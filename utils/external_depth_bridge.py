import torch
import sys
import os
from pathlib import Path
from typing import Dict, Optional, Tuple, List
import numpy as np

# Add paths
sys.path.append('/MonoGS/gaussian_splatting/')
sys.path.append('/EDGS/')

# Import MonoGS components
from scene.gaussian_model import GaussianModel
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import build_rotation

# Import EDGS components
sys.path.append('/EDGS/source')
from corr_init import init_gaussians_with_corr, init_gaussians_with_corr_fast
from utils_aux import log_samples
from losses import ssim, l1_loss, psnr
import lpips

# Import RoMa model
try:
    sys.path.append('/EDGS/submodules/RoMa')
    from romatch import roma_outdoor, roma_indoor
    HAS_ROMA = True
except ImportError:
    HAS_ROMA = False
    print("Warning: RoMa not available, using fast mode only")


# 全局函數，可以被 pickle
def patched_densification_for_edgs(self, new_xyz, new_features_dc, new_features_rest, 
                                   new_opacities, new_scaling, new_rotation, *args):
    """Global patched densification function for EDGS compatibility"""
    # EDGS 傳入 7 個參數，MonoGS 需要 8 個
    if len(args) == 1:  # EDGS 調用，只有 new_tmp_radii
        new_tmp_radii = args[0]
        # 創建預設的 kf_ids 和 n_obs
        n_points = len(new_xyz)
        new_kf_ids = torch.zeros(n_points, dtype=torch.int32)
        new_n_obs = torch.zeros(n_points, dtype=torch.int32)
        
        # 調用原始的 densification_postfix（需要從 self 獲取）
        return self._original_densification_postfix(
            new_xyz, new_features_dc, new_features_rest,
            new_opacities, new_scaling, new_rotation,
            new_kf_ids, new_n_obs
        )
    else:  # MonoGS 正常調用
        return self._original_densification_postfix(
            new_xyz, new_features_dc, new_features_rest,
            new_opacities, new_scaling, new_rotation,
            *args
        )


class EDGSBridge:
    """Bridge class to integrate EDGS fast initialization into MonoGS pipeline"""
    
    def __init__(self, 
                 gaussian_model: GaussianModel,
                 device: torch.device = torch.device('cuda'),
                 use_fast_init: bool = True):
        """Initialize the EDGS Bridge"""
        self.gaussian_model = gaussian_model
        self.device = device
        self.use_fast_init = use_fast_init
        self.lpips = lpips.LPIPS(net='vgg').to(device)
        self.initial_state = None
        self.roma_model = None  # 將在需要時初始化
        
    def _init_roma_model(self, model_type='outdoor'):
        """初始化 RoMa 模型"""
        if not HAS_ROMA:
            print("Warning: RoMa not available, falling back to fast mode")
            return None
            
        if self.roma_model is not None:
            return self.roma_model
            
        try:
            print(f"Initializing RoMa {model_type} model...")
            if model_type == 'indoor' or model_type == 'indoors':
                self.roma_model = roma_indoor(device=self.device)
            else:
                self.roma_model = roma_outdoor(device=self.device)
            
            # 重要：設置 RoMa 模型參數以避免尺寸不匹配
            # 選項 1：關閉 upsample 和 attenuate_cert
            self.roma_model.upsample_preds = False
            self.roma_model.symmetric = False
            self.roma_model.attenuate_cert = False  # 關閉 certainty attenuation 以避免尺寸不匹配
            
            print(f"RoMa {model_type} model initialized successfully")
            print(f"RoMa settings: upsample_preds={self.roma_model.upsample_preds}, "
                  f"symmetric={self.roma_model.symmetric}, "
                  f"attenuate_cert={self.roma_model.attenuate_cert}")
            return self.roma_model
        except Exception as e:
            print(f"Failed to initialize RoMa model: {e}")
            import traceback
            traceback.print_exc()
            return None
        
    def save_initial_state(self):
        """儲存高斯的初始狀態以便 EDGS 失敗時恢復"""
        self.initial_state = {}
        
        attrs_to_save = [
            '_xyz', '_features_dc', '_features_rest', '_scaling', 
            '_rotation', '_opacity', 'unique_kfIDs', 'n_obs', 
            'max_radii2D', 'xyz_gradient_accum', 'denom'
        ]
        
        for attr in attrs_to_save:
            if hasattr(self.gaussian_model, attr):
                value = getattr(self.gaussian_model, attr)
                if value is not None:
                    if isinstance(value, torch.Tensor):
                        self.initial_state[attr] = value.clone()
                    else:
                        self.initial_state[attr] = value
                        
        if hasattr(self.gaussian_model, 'spatial_lr_scale'):
            self.initial_state['spatial_lr_scale'] = self.gaussian_model.spatial_lr_scale
    
    def restore_initial_state(self):
        """恢復初始狀態"""
        if self.initial_state is None:
            return
            
        for key, value in self.initial_state.items():
            if hasattr(self.gaussian_model, key):
                if isinstance(value, torch.Tensor):
                    setattr(self.gaussian_model, key, value.clone())
                else:
                    setattr(self.gaussian_model, key, value)
    
    def _prepare_gaussian_model_for_edgs(self):
        """準備高斯模型以供 EDGS 使用"""
        with torch.no_grad():
            # 初始化 MonoGS 特有的屬性
            if not hasattr(self.gaussian_model, 'unique_kfIDs') or self.gaussian_model.unique_kfIDs is None:
                self.gaussian_model.unique_kfIDs = torch.zeros(0, dtype=torch.int32)
            else:
                if self.gaussian_model.unique_kfIDs.device.type != 'cpu':
                    self.gaussian_model.unique_kfIDs = self.gaussian_model.unique_kfIDs.cpu()
            
            if not hasattr(self.gaussian_model, 'n_obs') or self.gaussian_model.n_obs is None:
                self.gaussian_model.n_obs = torch.zeros(0, dtype=torch.int32)
            else:
                if self.gaussian_model.n_obs.device.type != 'cpu':
                    self.gaussian_model.n_obs = self.gaussian_model.n_obs.cpu()
            
            # 確保其他張量在 CUDA 上
            cuda_attrs = [
                '_xyz', '_features_dc', '_features_rest', '_scaling', 
                '_rotation', '_opacity', 'max_radii2D', 
                'xyz_gradient_accum', 'denom'
            ]
            
            for attr in cuda_attrs:
                if hasattr(self.gaussian_model, attr):
                    tensor = getattr(self.gaussian_model, attr)
                    if tensor is not None and isinstance(tensor, torch.Tensor):
                        if tensor.device != self.device:
                            setattr(self.gaussian_model, attr, tensor.to(self.device))
    
    def adapt_monogs_to_edgs_format(self, keyframe_data: Dict) -> Dict:
        """Adapt MonoGS keyframe data to EDGS expected format"""
        edgs_data = {}
        
        # 轉換相機參數
        if 'camera' in keyframe_data:
            camera = keyframe_data['camera']
            # 修正：確保從相機物件正確提取參數
            if hasattr(camera, 'FoVx'):
                edgs_data['FoVx'] = camera.FoVx
                edgs_data['FoVy'] = camera.FoVy
            else:
                edgs_data['FoVx'] = camera.get('FoVx', 1.0)
                edgs_data['FoVy'] = camera.get('FoVy', 1.0)
                
            if hasattr(camera, 'image_width'):
                edgs_data['image_width'] = camera.image_width
                edgs_data['image_height'] = camera.image_height
            else:
                edgs_data['image_width'] = camera.get('width', 640)
                edgs_data['image_height'] = camera.get('height', 480)
                
            # 添加相機內參
            if hasattr(camera, 'fx'):
                edgs_data['fx'] = camera.fx
                edgs_data['fy'] = camera.fy
                edgs_data['cx'] = camera.cx
                edgs_data['cy'] = camera.cy
            
        # 轉換位姿資訊
        if 'pose' in keyframe_data:
            pose = keyframe_data['pose']
            if isinstance(pose, np.ndarray):
                pose = torch.from_numpy(pose).float()
            elif isinstance(pose, torch.Tensor):
                pose = pose.float()
            pose = pose.to(self.device)
            edgs_data['world_view_transform'] = pose
            edgs_data['full_proj_transform'] = self._compute_projection_matrix(
                edgs_data, pose
            )
        elif 'camera' in keyframe_data and hasattr(keyframe_data['camera'], 'world_view_transform'):
            # 直接從相機物件獲取變換矩陣
            camera = keyframe_data['camera']
            edgs_data['world_view_transform'] = camera.world_view_transform
            edgs_data['full_proj_transform'] = camera.full_proj_transform
            
        # 轉換影像資料
        if 'image' in keyframe_data:
            image = keyframe_data['image']
        elif 'camera' in keyframe_data and hasattr(keyframe_data['camera'], 'original_image'):
            image = keyframe_data['camera'].original_image
        else:
            image = None
            
        if image is not None:
            if isinstance(image, np.ndarray):
                image = torch.from_numpy(image).float()
            elif isinstance(image, torch.Tensor):
                image = image.float()
            image = image.to(self.device)
            
            if image.dim() == 3 and image.shape[0] == 3:
                edgs_data['original_image'] = image
            elif image.dim() == 3 and image.shape[2] == 3:
                edgs_data['original_image'] = image.permute(2, 0, 1)
            else:
                edgs_data['original_image'] = image
                
        # 添加深度（如果有）
        if 'depth' in keyframe_data:
            depth = keyframe_data['depth']
            if isinstance(depth, np.ndarray):
                depth = torch.from_numpy(depth).float()
            elif isinstance(depth, torch.Tensor):
                depth = depth.float()
            depth = depth.to(self.device)
            edgs_data['depth'] = depth
            
        return edgs_data
    
    def _compute_projection_matrix(self, camera_data: Dict, pose: torch.Tensor) -> torch.Tensor:
        """Compute full projection matrix for EDGS"""
        fx = camera_data.get('fx', None)
        fy = camera_data.get('fy', None)
        
        if fx is None or fy is None:
            fx = camera_data['image_width'] / (2 * np.tan(camera_data['FoVx'] / 2))
            fy = camera_data['image_height'] / (2 * np.tan(camera_data['FoVy'] / 2))
        
        cx = camera_data.get('cx', camera_data['image_width'] / 2)
        cy = camera_data.get('cy', camera_data['image_height'] / 2)
        
        pose = pose.to(self.device)
        
        if pose.shape == (4, 4):
            znear = 0.01
            zfar = 100.0
            
            proj = torch.zeros(4, 4, device=self.device, dtype=torch.float32)
            proj[0, 0] = 2 * fx / camera_data['image_width']
            proj[1, 1] = 2 * fy / camera_data['image_height']
            proj[0, 2] = 2 * cx / camera_data['image_width'] - 1
            proj[1, 2] = 2 * cy / camera_data['image_height'] - 1
            proj[2, 2] = -(zfar + znear) / (zfar - znear)
            proj[2, 3] = -2 * zfar * znear / (zfar - znear)
            proj[3, 2] = -1
            
            full_proj = proj @ pose
            return full_proj
        else:
            raise ValueError(f"Unexpected pose shape: {pose.shape}")
    
    def _create_mock_scene(self, edgs_keyframes: List[Dict]):
        """Create a mock scene object that mimics EDGS scene structure"""
        class MockCamera:
            def __init__(self, kf, uid):
                device = kf.get('original_image').device if kf.get('original_image') is not None else torch.device('cuda')
                
                self.original_image = kf.get('original_image')
                self.world_view_transform = kf.get('world_view_transform')
                self.full_proj_transform = kf.get('full_proj_transform')
                self.image_width = kf.get('image_width', 640)
                self.image_height = kf.get('image_height', 480)
                self.FoVx = kf.get('FoVx', 1.0)
                self.FoVy = kf.get('FoVy', 1.0)
                self.uid = uid
                
                # 添加相機內參
                self.fx = kf.get('fx', self.image_width / (2 * np.tan(self.FoVx / 2)))
                self.fy = kf.get('fy', self.image_height / (2 * np.tan(self.FoVy / 2)))
                self.cx = kf.get('cx', self.image_width / 2)
                self.cy = kf.get('cy', self.image_height / 2)
                
                if self.world_view_transform is not None:
                    self.R = self.world_view_transform[:3, :3]
                    self.T = self.world_view_transform[:3, 3]
                    self.camera_center = -self.R.T @ self.T
                else:
                    self.R = None
                    self.T = None
                    self.camera_center = None
                
                self.depth = kf.get('depth', None)
                
        class MockScene:
            def __init__(self, keyframes):
                self.train_cameras = {}
                self.test_cameras = {}
                
                all_centers = []
                cameras = []
                
                for i, kf in enumerate(keyframes):
                    camera = MockCamera(kf, i)
                    cameras.append(camera)
                    if camera.camera_center is not None:
                        all_centers.append(camera.camera_center)
                
                self.train_cameras = cameras
                
                if all_centers:
                    all_centers = torch.stack(all_centers)
                    scene_center = all_centers.mean(dim=0)
                    dists = torch.norm(all_centers - scene_center, dim=1)
                    self.cameras_extent = dists.max().item()
                else:
                    self.cameras_extent = 1.0
                    
            def getTrainCameras(self):
                return self.train_cameras
                
            def getTestCameras(self):
                return self.test_cameras
                
        return MockScene(edgs_keyframes)
    
    def _setup_densification_patch(self):
        """設置 densification patch"""
        # 保存原始函數
        if not hasattr(self.gaussian_model, '_original_densification_postfix'):
            self.gaussian_model._original_densification_postfix = self.gaussian_model.densification_postfix
        
        # 使用 lambda 包裝以保持 self 引用
        self.gaussian_model.densification_postfix = lambda *args: patched_densification_for_edgs(
            self.gaussian_model, *args
        )
    
    def _restore_densification(self):
        """恢復原始 densification 函數"""
        if hasattr(self.gaussian_model, '_original_densification_postfix'):
            self.gaussian_model.densification_postfix = self.gaussian_model._original_densification_postfix
    
    def initialize_with_edgs(self, keyframes: List[Dict], init_config: Optional[Dict] = None) -> Dict:
        """Initialize Gaussians using EDGS - ADDITIVE mode"""
        if init_config is None:
            init_config = self._get_default_init_config()
        
        # Store current gaussians count
        n_gaussians_before = len(self.gaussian_model._xyz) if self.gaussian_model._xyz is not None else 0
        
        try:
            # Prepare gaussian model
            self._prepare_gaussian_model_for_edgs()
            
            # Convert keyframes to EDGS format
            edgs_keyframes = []
            for kf in keyframes:
                if hasattr(kf, 'original_image'):
                    edgs_kf = self.adapt_monogs_to_edgs_format({'camera': kf})
                else:
                    edgs_kf = self.adapt_monogs_to_edgs_format(kf)
                edgs_keyframes.append(edgs_kf)
            
            # Create mock scene
            mock_scene = self._create_mock_scene(edgs_keyframes)
            
            # Prepare args
            class Args:
                def __init__(self, config):
                    for key, value in config.items():
                        setattr(self, key, value)
            
            args = Args(init_config)
            
            # Setup densification patch
            self._setup_densification_patch()
            
            # Use RoMa if nns_per_ref > 1
            if init_config.get('nns_per_ref', 1) > 1 and not self.use_fast_init:
                # Initialize RoMa model
                roma_model_type = init_config.get('roma_model', 'outdoor')
                if roma_model_type in ['indoor', 'indoors']:
                    roma_model_type = 'indoor'
                
                roma_model = self._init_roma_model(roma_model_type)
                if roma_model is None:
                    Log("RoMa unavailable, using fast mode")
                    camera_set, selected_indices, visualization_dict = init_gaussians_with_corr_fast(
                        self.gaussian_model, mock_scene, args, self.device, 
                        verbose=init_config.get('verbose', False)
                    )
                else:
                    Log(f"Using RoMa {roma_model_type} for triangulation")
                    # Pass the RoMa model instance
                    camera_set, selected_indices, visualization_dict = init_gaussians_with_corr(
                        self.gaussian_model, mock_scene, args, self.device,
                        verbose=init_config.get('verbose', False),
                        roma_model=roma_model
                    )
            else:
                # Fast mode
                camera_set, selected_indices, visualization_dict = init_gaussians_with_corr_fast(
                    self.gaussian_model, mock_scene, args, self.device,
                    verbose=init_config.get('verbose', False)
                )
            
            # Restore densification
            self._restore_densification()
            
            # Ensure consistency
            n_gaussians_after = len(self.gaussian_model._xyz)
            n_added = n_gaussians_after - n_gaussians_before
            
            # Post-processing
            with torch.no_grad():
                self._ensure_device_consistency()
                
                # Only scale the NEW gaussians
                if n_added > 0:
                    scale_factor = init_config.get('scale_factor', 0.2)
                    self.gaussian_model._scaling[-n_added:] = self.gaussian_model.scaling_inverse_activation(
                        self.gaussian_model.scaling_activation(self.gaussian_model._scaling[-n_added:]) * scale_factor
                    )
            
            return {
                'n_splats_initial': n_gaussians_before,
                'n_splats_after': n_gaussians_after,
                'n_splats_added': n_added,
                'visualization': visualization_dict,
                'selected_cameras': selected_indices
            }
            
        except Exception as e:
            Log(f"EDGS initialization error: {e}")
            import traceback
            traceback.print_exc()
            # Don't restore - keep what we have
            self._restore_densification()
            raise
    
    def _ensure_device_consistency(self):
        """確保所有高斯模型張量在正確的設備上"""
        with torch.no_grad():
            # CUDA 張量
            cuda_attrs = [
                '_xyz', '_features_dc', '_features_rest', '_scaling', 
                '_rotation', '_opacity', 'max_radii2D', 
                'xyz_gradient_accum', 'denom'
            ]
            
            for attr in cuda_attrs:
                if hasattr(self.gaussian_model, attr):
                    tensor = getattr(self.gaussian_model, attr)
                    if tensor is not None and isinstance(tensor, torch.Tensor):
                        if tensor.device != self.device:
                            setattr(self.gaussian_model, attr, tensor.to(self.device))
            
            # CPU 張量（MonoGS 期望這些在 CPU）
            cpu_attrs = ['unique_kfIDs', 'n_obs']
            for attr in cpu_attrs:
                if hasattr(self.gaussian_model, attr):
                    tensor = getattr(self.gaussian_model, attr)
                    if tensor is not None and isinstance(tensor, torch.Tensor):
                        if tensor.device.type != 'cpu':
                            setattr(self.gaussian_model, attr, tensor.cpu())
    
    def _get_default_init_config(self) -> Dict:
        """Get default EDGS initialization configuration"""
        return {
            'use': True,
            'nns_per_ref': 3,  # Enable RoMa triangulation
            'num_refs': 5,    
            'matches_per_ref': 200,  
            'add_SfM_init': True,  # Add to existing points
            'scale_factor': 0.2,
            'scaling_factor': 0.005,
            'verbose': False,
            'proj_err_tolerance': 0.2,
            'roma_model': 'indoors',  # or 'outdoor'
        }
    
    def transfer_to_monogs(self) -> GaussianModel:
        """Transfer back to MonoGS - 確保所有屬性同步"""
        with torch.no_grad():
            # 檢查並修正 NaN 值
            if torch.isnan(self.gaussian_model._xyz).any():
                print("Warning: NaN values detected in positions")
                self.gaussian_model._xyz = torch.nan_to_num(self.gaussian_model._xyz)
            
            # 確保參數在合理範圍內
            self.gaussian_model._opacity = torch.clamp(
                self.gaussian_model._opacity, 
                min=-10.0, 
                max=10.0
            )
            
            self.gaussian_model._scaling = torch.clamp(
                self.gaussian_model._scaling,
                min=-10.0,
                max=1.0
            )
            
            # 檢查旋轉四元數
            if hasattr(self.gaussian_model, '_rotation'):
                norm = torch.norm(self.gaussian_model._rotation, dim=1, keepdim=True)
                self.gaussian_model._rotation = self.gaussian_model._rotation / (norm + 1e-8)
            
            # 同步 n_obs 和 unique_kfIDs 的大小
            current_n_points = len(self.gaussian_model._xyz)
            
            # 檢查並修正 unique_kfIDs
            if not hasattr(self.gaussian_model, 'unique_kfIDs') or self.gaussian_model.unique_kfIDs is None:
                self.gaussian_model.unique_kfIDs = torch.zeros(current_n_points, dtype=torch.int32)
            elif len(self.gaussian_model.unique_kfIDs) != current_n_points:
                print(f"Resizing unique_kfIDs from {len(self.gaussian_model.unique_kfIDs)} to {current_n_points}")
                if len(self.gaussian_model.unique_kfIDs) < current_n_points:
                    padding = torch.zeros(
                        current_n_points - len(self.gaussian_model.unique_kfIDs), 
                        dtype=torch.int32
                    )
                    self.gaussian_model.unique_kfIDs = torch.cat([
                        self.gaussian_model.unique_kfIDs, 
                        padding
                    ])
                else:
                    self.gaussian_model.unique_kfIDs = self.gaussian_model.unique_kfIDs[:current_n_points]
            
            # 檢查並修正 n_obs
            if not hasattr(self.gaussian_model, 'n_obs') or self.gaussian_model.n_obs is None:
                self.gaussian_model.n_obs = torch.zeros(current_n_points, dtype=torch.int32)
            elif len(self.gaussian_model.n_obs) != current_n_points:
                print(f"Resizing n_obs from {len(self.gaussian_model.n_obs)} to {current_n_points}")
                if len(self.gaussian_model.n_obs) < current_n_points:
                    padding = torch.zeros(
                        current_n_points - len(self.gaussian_model.n_obs), 
                        dtype=torch.int32
                    )
                    self.gaussian_model.n_obs = torch.cat([
                        self.gaussian_model.n_obs, 
                        padding
                    ])
                else:
                    self.gaussian_model.n_obs = self.gaussian_model.n_obs[:current_n_points]
            
            # 最終設備一致性檢查
            self._ensure_device_consistency()
            
            # 確保 xyz_gradient_accum 和 denom 也同步
            if hasattr(self.gaussian_model, 'xyz_gradient_accum'):
                if self.gaussian_model.xyz_gradient_accum is None or len(self.gaussian_model.xyz_gradient_accum) != current_n_points:
                    self.gaussian_model.xyz_gradient_accum = torch.zeros((current_n_points, 1), device=self.device)
            
            if hasattr(self.gaussian_model, 'denom'):
                if self.gaussian_model.denom is None or len(self.gaussian_model.denom) != current_n_points:
                    self.gaussian_model.denom = torch.zeros((current_n_points, 1), device=self.device)
            
            if hasattr(self.gaussian_model, 'max_radii2D'):
                if self.gaussian_model.max_radii2D is None or len(self.gaussian_model.max_radii2D) != current_n_points:
                    self.gaussian_model.max_radii2D = torch.zeros(current_n_points, device=self.device)
        
        return self.gaussian_model


# Utility function
def initialize_monogs_with_edgs(gaussian_model: GaussianModel,
                               keyframes: List[Dict],
                               config: Optional[Dict] = None,
                               device: torch.device = torch.device('cuda')) -> Tuple[GaussianModel, Dict]:
    """
    Convenience function to initialize MonoGS Gaussians with EDGS
    
    Args:
        gaussian_model: MonoGS GaussianModel instance
        keyframes: List of keyframes (can be camera objects or dicts)
        config: EDGS configuration dict
        device: torch device
    
    Returns:
        Tuple of (updated gaussian_model, stats dict)
    """
    bridge = EDGSBridge(gaussian_model, device)
    stats = bridge.initialize_with_edgs(keyframes, config)
    return bridge.transfer_to_monogs(), stats