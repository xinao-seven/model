"""
三维地质建模脚本 - 基于GemPy和钻孔数据

功能：使用GemPy进行三维地质建模
数据：钻孔collar、deviation、lithology数据
作者：自动生成
日期：2025-11-10

依赖库版本：
    - gempy >= 3.0.0
    - gempy-viewer >= 2025.1.0
    - pandas >= 2.0.0
    - numpy >= 1.24.0
    - subsurface >= 0.3.0
    - wellpathpy >= 0.5.0
    - pyvista >= 0.43.0
"""

# ==================== 导入所需库 ====================
import os
import numpy as np
import pandas as pd
from typing import Tuple, Dict, List

# GemPy相关
import gempy as gp
import gempy_viewer as gpv

# Subsurface相关（用于处理钻孔数据）
from subsurface.core.geological_formats import Collars, Survey, BoreholeSet
from subsurface.core.geological_formats.boreholes._combine_trajectories import MergeOptions
from subsurface.core.reader_helpers.readers_data import GenericReaderFilesHelper
from subsurface.core.structs.base_structures.base_structures_enum import SpecialCellCase
from subsurface.modules.reader.wells.read_borehole_interface import read_collar, read_survey, read_lith
from subsurface.modules.visualization import to_pyvista_line, to_pyvista_points, pv_plot
import subsurface as ss

# ==================== 配置参数区域 ====================
class ModelConfig:
    """模型配置参数类 - 所有可调整参数集中在此"""
    
    # 数据文件路径
    DATA_DIR = r"d:\axinao\model\src_Implicit\data"
    COLLAR_FILE = "collars.csv"
    DEVIATION_FILE = "deviation.csv"
    LITHOLOGY_FILE = "lithology.csv"
    
    # 模型参数
    PROJECT_NAME = "Borehole_Geological_Model"
    
    # 网格分辨率 (x, y, z) - 可调整以平衡精度和计算速度
    GRID_RESOLUTION = (100, 100, 100)
    
    # 插值参数
    INTERPOLATION_RANGE = 5.0  # 插值范围
    C_O = 10.0  # 协方差参数
    MESH_EXTRACTION = True  # 是否提取网格
    NUMBER_OCTREE_LEVELS = 3  # 八叉树层级
    
    # 可视化参数
    SHOW_3D_PLOT = True
    SHOW_2D_PLOT = True
    IMAGE_MODE = False  # 是否保存为图片
    SAVE_IMAGES = True  # 是否保存图片到文件
    OUTPUT_DIR = r"d:\axinao\model\src_Implicit\output"  # 输出目录
    Z_SCALE = 10.0  # Z轴缩放比例（垂直夸大系数）
    
    # 3D可视化细节参数
    SHOW_BOREHOLE_POINTS = False  # 是否显示钻孔数据点
    SHOW_LAYER_SLICES = False  # 是否显示地层切片
    SHOW_LAYER_VOLUME = True  # 是否显示完整地层体积模型
    LAYER_OPACITY = 0.8  # 地层透明度 (0-1)
    SHOW_LAYER_EDGES = False  # 是否显示地层边缘
    
    # 模型导出参数
    EXPORT_MODEL = True  # 是否导出模型
    EXPORT_FORMATS = ['vtk', 'vtu', 'stl']  # 导出格式: 'vtk', 'vtu', 'stl', 'obj', 'ply'
    EXPORT_SEPARATE_LAYERS = True  # 是否分层导出（每个地层单独一个文件）
    EXPORT_COMBINED_MODEL = True  # 是否导出合并模型（所有地层在一个文件中）
    
    # 地表数据配置
    USE_SURFACE_TOPOGRAPHY = True  # 是否使用地表地形
    SURFACE_DATA_FILE = "surface_points.csv"  # 地表点数据文件 (x, y, z)
    SURFACE_RESOLUTION = 50  # 地表网格分辨率
    CREATE_FLAT_SURFACE = False  # 如果没有地表数据，是否创建平面地表
    SURFACE_ELEVATION = None  # 平面地表高程（None则自动从collar数据推断）
    
    # 地层单元定义（可根据实际数据调整）
    LITHOLOGY_UNITS = {
        "loss": {"id": 1, "description": "损失层"},
        "lishi": {"id": 2, "description": "砾石层"},
        "coaltop": {"id": 3, "description": "煤层顶部"},
        "mix": {"id": 4, "description": "混合层"},
        "coalmid": {"id": 5, "description": "煤层中部"},
        "stonetop": {"id": 6, "description": "石层顶部"},
        "coalend": {"id": 7, "description": "煤层底部"},
        "stonemid": {"id": 8, "description": "石层中部"},
        "coalendo": {"id": 9, "description": "煤层末端"},
        "stoneend": {"id": 10, "description": "石层底部"},
    }


# ==================== 数据读取函数 ====================

def load_collar_data(data_dir: str, filename: str) -> pd.DataFrame:
    """
    读取钻孔孔口坐标数据
    
    Args:
        data_dir (str): 数据文件夹路径
        filename (str): collar文件名
    
    Returns:
        pd.DataFrame: 包含hole_id, x, y, z的DataFrame
    
    Raises:
        FileNotFoundError: 如果文件不存在
        ValueError: 如果数据格式不正确
    """
    filepath = os.path.join(data_dir, filename)
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Collar文件未找到: {filepath}")
    
    try:
        df = pd.read_csv(filepath)
        
        # 验证必需列
        required_cols = ['hole_id', 'x', 'y', 'z']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"Collar文件缺少必需列。需要: {required_cols}, 实际: {df.columns.tolist()}")
        
        # 清理空格
        df['hole_id'] = df['hole_id'].astype(str).str.strip()
        df['x'] = pd.to_numeric(df['x'], errors='coerce')
        df['y'] = pd.to_numeric(df['y'], errors='coerce')
        df['z'] = pd.to_numeric(df['z'], errors='coerce')
        
        # 删除包含NaN的行
        df = df.dropna()
        
        print(f"成功读取 {len(df)} 个钻孔的collar数据")
        return df
        
    except Exception as e:
        raise ValueError(f"读取collar数据时出错: {str(e)}")


def load_deviation_data(data_dir: str, filename: str) -> pd.DataFrame:
    """
    读取钻孔偏斜数据
    
    Args:
        data_dir (str): 数据文件夹路径
        filename (str): deviation文件名
    
    Returns:
        pd.DataFrame: 包含hole_id, md, inc, az的DataFrame
    
    Notes:
        - md: 测量深度(米)
        - inc: 倾角(度，自竖直方向)
        - az: 方位角(度，北起顺时针)
    """
    filepath = os.path.join(data_dir, filename)
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Deviation文件未找到: {filepath}")
    
    try:
        df = pd.read_csv(filepath)
        
        # 验证必需列
        required_cols = ['hole_id', 'md', 'inc', 'az']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"Deviation文件缺少必需列。需要: {required_cols}")
        
        # 数据清理
        df['hole_id'] = df['hole_id'].astype(str).str.strip()
        df['md'] = pd.to_numeric(df['md'], errors='coerce')
        df['inc'] = pd.to_numeric(df['inc'], errors='coerce')
        df['az'] = pd.to_numeric(df['az'], errors='coerce')
        
        df = df.dropna()
        
        print(f"成功读取 {len(df)} 条deviation数据")
        return df
        
    except Exception as e:
        raise ValueError(f"读取deviation数据时出错: {str(e)}")


def load_lithology_data(data_dir: str, filename: str) -> pd.DataFrame:
    """
    读取岩性/层位区间数据
    
    Args:
        data_dir (str): 数据文件夹路径
        filename (str): lithology文件名
    
    Returns:
        pd.DataFrame: 包含hole_id, from_md, to_md, unit的DataFrame
    
    Notes:
        - from_md: 起始深度
        - to_md: 结束深度
        - unit: 地层/岩性名称
    """
    filepath = os.path.join(data_dir, filename)
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Lithology文件未找到: {filepath}")
    
    try:
        df = pd.read_csv(filepath)
        
        # 验证必需列
        required_cols = ['hole_id', 'from_md', 'to_md', 'unit']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"Lithology文件缺少必需列。需要: {required_cols}")
        
        # 数据清理
        df['hole_id'] = df['hole_id'].astype(str).str.strip()
        df['from_md'] = pd.to_numeric(df['from_md'], errors='coerce')
        df['to_md'] = pd.to_numeric(df['to_md'], errors='coerce')
        df['unit'] = df['unit'].astype(str).str.strip()
        
        df = df.dropna()
        
        print(f"成功读取 {len(df)} 条lithology数据")
        print(f"包含的地层单元: {df['unit'].unique().tolist()}")
        return df
        
    except Exception as e:
        raise ValueError(f"读取lithology数据时出错: {str(e)}")


def load_surface_data(data_dir: str, filename: str) -> pd.DataFrame:
    """
    读取地表高程数据
    
    Args:
        data_dir (str): 数据文件夹路径
        filename (str): 地表数据文件名
    
    Returns:
        pd.DataFrame: 包含x, y, z的DataFrame，如果文件不存在则返回None
    
    注意：
        如果没有地表数据文件，可以：
        1. 从DEM数据提取
        2. 使用collar点的z坐标作为地表
        3. 使用固定高程的平面
    """
    filepath = os.path.join(data_dir, filename)
    
    if not os.path.exists(filepath):
        print(f"  地表数据文件不存在: {filepath}")
        return None
    
    try:
        df = pd.read_csv(filepath)
        
        # 验证必需列
        required_cols = ['x', 'y', 'z']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"地表文件缺少必需列。需要: {required_cols}")
        
        # 数据清理
        df['x'] = pd.to_numeric(df['x'], errors='coerce')
        df['y'] = pd.to_numeric(df['y'], errors='coerce')
        df['z'] = pd.to_numeric(df['z'], errors='coerce')
        df = df.dropna()
        
        print(f"  成功读取 {len(df)} 个地表点")
        return df
        
    except Exception as e:
        print(f"  读取地表数据时出错: {str(e)}")
        return None


def create_surface_points_from_collar(collar_df: pd.DataFrame, 
                                      config) -> pd.DataFrame:
    """
    从collar数据创建地表点
    （假设collar点位于地表或接近地表）
    
    Args:
        collar_df (pd.DataFrame): collar数据
        config: 配置参数
    
    Returns:
        pd.DataFrame: 地表点数据
    """
    print("  - 从collar数据推断地表...")
    
    surface_points = collar_df[['x', 'y', 'z']].copy()
    
    # 如果指定了固定高程
    if config.SURFACE_ELEVATION is not None:
        surface_points['z'] = config.SURFACE_ELEVATION
        print(f"    使用固定高程: {config.SURFACE_ELEVATION}")
    else:
        print(f"    使用collar点高程作为地表")
    
    print(f"    生成 {len(surface_points)} 个地表点")
    return surface_points


def interpolate_surface_grid(surface_points: pd.DataFrame, 
                            extent: list, 
                            resolution: int = 50) -> pd.DataFrame:
    """
    对地表点进行网格化插值
    
    Args:
        surface_points (pd.DataFrame): 离散地表点
        extent (list): 模型范围 [xmin, xmax, ymin, ymax, zmin, zmax]
        resolution (int): 网格分辨率
    
    Returns:
        pd.DataFrame: 规则网格的地表点
    """
    from scipy.interpolate import griddata
    
    print(f"  - 对地表进行网格插值（分辨率: {resolution}x{resolution}）...")
    
    # 创建规则网格
    x_grid = np.linspace(extent[0], extent[1], resolution)
    y_grid = np.linspace(extent[2], extent[3], resolution)
    X, Y = np.meshgrid(x_grid, y_grid)
    
    # 插值地表高程
    points = surface_points[['x', 'y']].values
    values = surface_points['z'].values
    
    Z = griddata(points, values, (X, Y), method='cubic')
    
    # 处理NaN值（使用最近邻插值）
    if np.any(np.isnan(Z)):
        print("    检测到插值空白区域，使用最近邻填充...")
        Z_nearest = griddata(points, values, (X, Y), method='nearest')
        Z = np.where(np.isnan(Z), Z_nearest, Z)
    
    # 转换为DataFrame
    grid_points = pd.DataFrame({
        'x': X.ravel(),
        'y': Y.ravel(),
        'z': Z.ravel()
    })
    
    print(f"    生成 {len(grid_points)} 个网格化地表点")
    return grid_points


# ==================== 钻孔数据处理函数 ====================

def create_structural_elements_from_data(collar_df: pd.DataFrame, 
                                         lithology_df: pd.DataFrame,
                                         surface_df: pd.DataFrame = None,
                                         config = None) -> tuple:
    """
    从钻孔数据创建GemPy结构元素（包含地表层，不使用deviation数据，假设垂直钻孔）
    
    Args:
        collar_df (pd.DataFrame): collar数据
        lithology_df (pd.DataFrame): lithology数据
        surface_df (pd.DataFrame): 地表点数据（可选）
        config: 配置参数
    
    Returns:
        tuple: (structural_elements列表, 岩性单元映射字典, 模型范围)
    """
    print("\n正在从钻孔数据创建结构元素...")
    
    # 1. 获取所有唯一的岩性单元
    unique_units = sorted(lithology_df['unit'].unique())
    print(f"  - 发现 {len(unique_units)} 种岩性单元: {unique_units}")
    
    # 2. 创建颜色生成器
    colors_generator = gp.data.ColorsGenerator()
    
    # 3. 创建结构元素列表
    elements = []
    elements_dict = {}
    
    # 4. 首先添加地表层（如果提供）
    if surface_df is not None and len(surface_df) > 0:
        print("\n  [添加地表层]")
        surface_id = 0  # 地表层ID为0
        surface_color = '#8B4513'  # 棕色
        
        # 地表中心点（用于生成方位）
        center_x = surface_df['x'].mean()
        center_y = surface_df['y'].mean()
        center_z = surface_df['z'].mean()
        
        # 地表方位（法向量向下，因为地表是最顶层）
        surface_orientations = gp.data.OrientationsTable.from_arrays(
            x=np.array([center_x]),
            y=np.array([center_y]),
            z=np.array([center_z]),
            G_x=np.array([0.0]),
            G_y=np.array([0.0]),
            G_z=np.array([-1.0]),  # 向下
            names=['surface'],
            name_id_map={'surface': surface_id}
        )
        
        # 创建地表结构元素
        surface_element = gp.data.StructuralElement(
            name='surface',
            id=surface_id,
            color=surface_color,
            surface_points=gp.data.SurfacePointsTable.from_arrays(
                x=surface_df['x'].values,
                y=surface_df['y'].values,
                z=surface_df['z'].values,
                names=['surface'] * len(surface_df),
                name_id_map={'surface': surface_id}
            ),
            orientations=surface_orientations
        )
        
        elements.append(surface_element)
        elements_dict['surface'] = {'id': surface_id, 'color': surface_color}
        print(f"  ✓ surface: {len(surface_df)} 个地表点 + 1 个方位点")
    
    # 5. 为每个岩性单元创建结构元素
    print("\n  [添加地层单元]")
    for idx, unit_name in enumerate(unique_units):
        unit_id = idx + 1  # 从1开始，0留给地表
        color = next(colors_generator)
        
        # 获取该岩性单元的所有界面点
        unit_data = lithology_df[lithology_df['unit'] == unit_name]
        
        # 收集表面点坐标（岩性单元的顶部）
        surface_points_list = []
        
        for _, row in unit_data.iterrows():
            hole_id = row['hole_id']
            from_depth = row['from_md']
            
            # 获取该钻孔的collar坐标
            collar_row = collar_df[collar_df['hole_id'] == hole_id]
            if len(collar_row) == 0:
                continue
                
            collar_x = collar_row['x'].values[0]
            collar_y = collar_row['y'].values[0]
            collar_z = collar_row['z'].values[0]
            
            # 假设垂直钻孔，表面点的z坐标 = collar_z - from_depth
            point_z = collar_z - from_depth
            
            surface_points_list.append({
                'x': collar_x,
                'y': collar_y,
                'z': point_z
            })
        
        if len(surface_points_list) == 0:
            print(f"  警告: 岩性单元 '{unit_name}' 没有有效的表面点，跳过")
            continue
        
        # 转换为numpy数组
        points_array = pd.DataFrame(surface_points_list)
        
        # 从表面点创建方位数据（假设水平层）
        # 取中心点作为方位点
        center_x = points_array['x'].mean()
        center_y = points_array['y'].mean()
        center_z = points_array['z'].mean()
        
        # 创建方位数据（水平层的法向量指向上方：[0, 0, 1]）
        orientations = gp.data.OrientationsTable.from_arrays(
            x=np.array([center_x]),
            y=np.array([center_y]),
            z=np.array([center_z]),
            G_x=np.array([0.0]),
            G_y=np.array([0.0]),
            G_z=np.array([1.0]),
            names=[unit_name],
            name_id_map={unit_name: unit_id}
        )
        
        # 创建结构元素
        element = gp.data.StructuralElement(
            name=unit_name,
            id=unit_id,
            color=color,
            surface_points=gp.data.SurfacePointsTable.from_arrays(
                x=points_array['x'].values,
                y=points_array['y'].values,
                z=points_array['z'].values,
                names=[unit_name] * len(points_array),
                name_id_map={unit_name: unit_id}
            ),
            orientations=orientations
        )
        
        elements.append(element)
        elements_dict[unit_name] = {'id': unit_id, 'color': color}
        
        print(f"  ✓ {unit_name}: {len(points_array)} 个表面点 + 1 个方位点")
    
    # 4. 计算模型范围
    all_x = []
    all_y = []
    all_z = []
    
    for element in elements:
        coords = element.surface_points.xyz
        all_x.extend(coords[:, 0])
        all_y.extend(coords[:, 1])
        all_z.extend(coords[:, 2])
    
    # 添加一些边界缓冲
    x_range = max(all_x) - min(all_x)
    y_range = max(all_y) - min(all_y)
    z_range = max(all_z) - min(all_z)
    
    extent = [
        min(all_x) - x_range * 0.1,
        max(all_x) + x_range * 0.1,
        min(all_y) - y_range * 0.1,
        max(all_y) + y_range * 0.1,
        min(all_z) - z_range * 0.1,
        max(all_z) + z_range * 0.1
    ]
    
    print(f"\n  模型范围计算完成:")
    print(f"    X: [{extent[0]:.2f}, {extent[1]:.2f}]")
    print(f"    Y: [{extent[2]:.2f}, {extent[3]:.2f}]")
    print(f"    Z: [{extent[4]:.2f}, {extent[5]:.2f}]")
    
    return elements, elements_dict, extent


def create_gempy_model(elements: list, extent: list, config: ModelConfig) -> gp.data.GeoModel:
    """
    创建GemPy地质模型
    
    Args:
        elements (list): 结构元素列表
        extent (list): 模型范围 [xmin, xmax, ymin, ymax, zmin, zmax]
        config (ModelConfig): 配置参数
    
    Returns:
        gp.data.GeoModel: GemPy地质模型对象
    """
    print("\n正在创建GemPy地质模型...")
    
    # 1. 创建结构组
    group = gp.data.StructuralGroup(
        name="Stratigraphic_Pile",
        elements=elements,
        structural_relation=gp.data.StackRelationType.ERODE
    )
    
    # 2. 创建结构框架
    structural_frame = gp.data.StructuralFrame(
        structural_groups=[group],
        color_gen=gp.data.ColorsGenerator()
    )
    
    print(f"  - 结构框架创建完成，包含 {len(elements)} 个地层单元")
    
    # 3. 创建GeoModel
    geo_model = gp.data.GeoModel(
        name=config.PROJECT_NAME,
        structural_frame=structural_frame,
        grid=gp.data.Grid(
            extent=extent,
            resolution=config.GRID_RESOLUTION
        ),
        interpolation_options=gp.data.InterpolationOptions(
            range=config.INTERPOLATION_RANGE,
            c_o=config.C_O,
            mesh_extraction=config.MESH_EXTRACTION,
            number_octree_levels=config.NUMBER_OCTREE_LEVELS,
        )
    )
    
    print(f"  - GeoModel创建完成")
    print(f"  - 网格分辨率: {config.GRID_RESOLUTION}")
    
    return geo_model


def compute_and_visualize_model(geo_model: gp.data.GeoModel, config: ModelConfig):
    """
    计算并可视化地质模型
    
    Args:
        geo_model (gp.data.GeoModel): GemPy地质模型
        config (ModelConfig): 配置参数
    """
    print("\n正在计算地质模型...")
    
    # 确保输出目录存在
    if config.SAVE_IMAGES:
        os.makedirs(config.OUTPUT_DIR, exist_ok=True)
        print(f"  输出目录: {config.OUTPUT_DIR}")
    
    # 计算模型
    gp.compute_model(
        geo_model,
        engine_config=gp.data.GemPyEngineConfig(
            backend=gp.data.AvailableBackends.numpy,
            use_gpu=False,
            dtype='float64'
        )
    )
    
    print("  ✓ 模型计算完成！")
    
    # 2D可视化
    if config.SHOW_2D_PLOT:
        print("\n正在生成2D可视化...")
        try:
            import matplotlib.pyplot as plt
            
            # 创建2D图
            p2d = gpv.plot_2d(geo_model, show_data=True, direction='y')
            
            # 保存图片
            if config.SAVE_IMAGES:
                output_path = os.path.join(config.OUTPUT_DIR, "model_2d_section.png")
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                print(f"  ✓ 2D图已保存: {output_path}")
            
            plt.show()
            print("  ✓ 2D图生成完成")
        except Exception as e:
            print(f"  2D可视化失败: {str(e)}")
    
    # 3D可视化
    if config.SHOW_3D_PLOT:
        print("\n正在生成3D可视化...")
        try:
            import pyvista as pv
            
            # 创建3D可视化
            plotter = pv.Plotter(window_size=[1920, 1080])
            plotter.set_background('white')
            
            # 尝试添加地层模型网格
            layer_mesh_added = False
            try:
                if hasattr(geo_model, 'solutions') and hasattr(geo_model.solutions, 'raw_arrays'):
                    lith_block = geo_model.solutions.raw_arrays.litho_faults_block
                    grid = geo_model.grid.regular_grid
                    
                    print(f"  - 地层数据形状: {lith_block.shape}")
                    print(f"  - 网格分辨率: {grid.resolution}")
                    print(f"  - Z轴缩放比例: {config.Z_SCALE}x")
                    
                    # 创建结构化网格（应用Z轴缩放）
                    grid_3d = pv.ImageData(
                        dimensions=grid.resolution,
                        spacing=(
                            (grid.extent[1] - grid.extent[0]) / (grid.resolution[0]),
                            (grid.extent[3] - grid.extent[2]) / (grid.resolution[1]),
                            (grid.extent[5] - grid.extent[4]) / (grid.resolution[2]) * config.Z_SCALE  # Z轴放大
                        ),
                        origin=(grid.extent[0], grid.extent[2], grid.extent[4] * config.Z_SCALE)  # Z轴原点也缩放
                    )
                    
                    # 添加岩性数据
                    grid_3d['lithology'] = lith_block.ravel()
                    
                    # 根据配置选择显示方式
                    if config.SHOW_LAYER_SLICES:
                        # 方法1: 显示正交切片
                        slices = grid_3d.slice_orthogonal()
                        plotter.add_mesh(
                            slices,
                            scalars='lithology',
                            cmap='tab20',
                            opacity=0.9,
                            show_scalar_bar=True,
                            scalar_bar_args={'title': '地层编号', 'vertical': True},
                            show_edges=False
                        )
                        print("  ✓ 地层模型切片已添加")
                    
                    if config.SHOW_LAYER_VOLUME:
                        # 方法2: 显示完整的3D体积模型
                        # 使用threshold提取每个地层并分别显示
                        unique_lith = np.unique(lith_block)
                        unique_lith = unique_lith[unique_lith > 0]  # 排除basement
                        
                        print(f"  - 正在生成 {len(unique_lith)} 个地层的3D模型...")
                        
                        for lith_id in unique_lith:
                            # 提取单个地层
                            layer = grid_3d.threshold([lith_id - 0.5, lith_id + 0.5], scalars='lithology')
                            
                            if layer.n_points > 0:
                                # 提取地层表面
                                layer_surface = layer.extract_surface()
                                
                                # 获取对应的颜色
                                element_idx = int(lith_id) - 1
                                if element_idx < len(geo_model.structural_frame.structural_elements):
                                    element_color = geo_model.structural_frame.structural_elements[element_idx].color
                                else:
                                    element_color = 'gray'
                                
                                plotter.add_mesh(
                                    layer_surface,
                                    color=element_color,
                                    opacity=config.LAYER_OPACITY,
                                    show_edges=config.SHOW_LAYER_EDGES,
                                    edge_color='black',
                                    line_width=0.5,
                                    smooth_shading=True
                                )
                        
                        print(f"  ✓ 完整地层3D模型已添加")
                    
                    layer_mesh_added = True
                    
            except Exception as e:
                print(f"  ! 地层网格添加失败: {str(e)}")
                print("    将只显示钻孔数据点")
                import traceback
                traceback.print_exc()
            
            # 添加钻孔数据点（根据配置决定是否显示）
            legend_entries = []
            if config.SHOW_BOREHOLE_POINTS:
                for element in geo_model.structural_frame.structural_elements:
                    if len(element.surface_points.data) > 0:
                        points = element.surface_points.xyz.copy()
                        # 对Z坐标应用缩放
                        points[:, 2] *= config.Z_SCALE
                        point_cloud = pv.PolyData(points)
                        
                        plotter.add_mesh(
                            point_cloud, 
                            color=element.color,
                            point_size=25,
                            render_points_as_spheres=True,
                            label=element.name
                        )
                        legend_entries.append([element.name, element.color])
                
                print(f"  ✓ 钻孔数据点已添加 ({len(legend_entries)} 个地层)")
            else:
                # 即使不显示点，也创建图例
                for element in geo_model.structural_frame.structural_elements:
                    legend_entries.append([element.name, element.color])
                print(f"  - 钻孔数据点已隐藏（可在配置中启用）")
            
            # 添加图例
            if legend_entries:
                plotter.add_legend(legend_entries, bcolor='white', face='rectangle', size=(0.25, 0.3))
            
            # 添加坐标轴和边界框
            plotter.show_axes()
            plotter.add_bounding_box(color='black', line_width=2)
            
            # 设置相机视角
            plotter.camera_position = 'iso'
            plotter.camera.zoom(1.0)
            
            # 添加标题
            title_parts = [f"{config.PROJECT_NAME} - 三维地质模型"]
            if config.SHOW_LAYER_VOLUME:
                title_parts.append("完整地层体积模型")
            elif config.SHOW_LAYER_SLICES:
                title_parts.append("地层切片模型")
            if config.SHOW_BOREHOLE_POINTS:
                title_parts.append("+ 钻孔数据点")
            title_parts.append(f"(Z轴放大 {config.Z_SCALE}x)")
            
            title = "\n".join(title_parts)
            
            plotter.add_text(
                title, 
                position='upper_edge',
                font_size=14,
                color='black',
                font='arial'
            )
            
            # 显示交互式3D窗口（用户可以在窗口中手动截图）
            print("\n  正在打开交互式3D窗口...")
            print("  提示: 可以用鼠标旋转、缩放模型")
            
            if config.SAVE_IMAGES:
                output_path = os.path.join(config.OUTPUT_DIR, "model_3d_complete.png")
                print(f"  提示: 窗口打开后，截图将自动保存到: {output_path}")
                print(f"  提示: 您可以调整视角后关闭窗口，截图会自动保存")
                
                # 显示窗口，关闭时自动截图
                plotter.show(screenshot=output_path)
                print(f"  ✓ 3D模型截图已保存: {output_path}")
            else:
                # 不保存截图，只显示
                plotter.show()
            
            print(f"  ✓ 3D可视化完成")
                
        except Exception as e:
            print(f"  3D可视化失败: {str(e)}")
            print("  提示: 2D可视化已成功生成")
            import traceback
            traceback.print_exc()
            import traceback
            traceback.print_exc()


def export_model(geo_model, config: ModelConfig):
    """
    导出地质模型为多种格式
    
    Args:
        geo_model: GemPy地质模型对象
        config: 模型配置参数
    
    支持的导出格式:
        - vtk: VTK Legacy格式（可用ParaView打开）
        - vtu: VTK XML格式（可用ParaView打开）
        - stl: STL格式（可用于3D打印）
        - obj: OBJ格式（可用Blender等软件打开）
        - ply: PLY格式（点云格式）
    """
    import pyvista as pv
    
    if not config.EXPORT_MODEL:
        print("\n[跳过] 模型导出功能已关闭")
        return
    
    print("\n" + "="*60)
    print("[步骤5] 导出地质模型")
    print("="*60)
    
    # 确保输出目录存在
    export_dir = os.path.join(config.OUTPUT_DIR, "exported_models")
    os.makedirs(export_dir, exist_ok=True)
    
    try:
        # 获取模型解决方案
        solution = geo_model.solutions.raw_arrays
        scalar_field = solution.scalar_field_matrix
        
        # 获取网格参数
        grid = geo_model.grid.regular_grid
        extent = grid.extent
        resolution = grid.resolution
        
        # 创建PyVista网格
        grid_3d = scalar_field.reshape(*resolution)
        
        # 应用Z轴缩放
        spacing = [
            (extent[1] - extent[0]) / resolution[0],
            (extent[3] - extent[2]) / resolution[1],
            (extent[5] - extent[4]) / resolution[2] * config.Z_SCALE
        ]
        origin = [extent[0], extent[2], extent[4] * config.Z_SCALE]
        
        # 创建基础网格
        image_data = pv.ImageData(
            dimensions=resolution,
            spacing=spacing,
            origin=origin
        )
        image_data["geological_formation"] = grid_3d.flatten(order='F')
        
        # 获取地层信息
        structural_frame = geo_model.structural_frame
        element_names = [elem.name for elem in structural_frame.structural_elements]
        n_elements = len(element_names)
        
        print(f"\n检测到 {n_elements} 个地层单元: {element_names}")
        print(f"导出格式: {', '.join(config.EXPORT_FORMATS)}")
        print(f"输出目录: {export_dir}")
        
        exported_files = []
        
        # 导出合并模型
        if config.EXPORT_COMBINED_MODEL:
            print("\n1. 导出合并模型（所有地层）...")
            
            for fmt in config.EXPORT_FORMATS:
                try:
                    filename = f"geological_model_combined.{fmt}"
                    filepath = os.path.join(export_dir, filename)
                    
                    if fmt in ['vtk', 'vtu']:
                        image_data.save(filepath)
                    elif fmt == 'stl':
                        # STL需要surface mesh
                        surface = image_data.extract_surface()
                        surface.save(filepath)
                    elif fmt == 'obj':
                        surface = image_data.extract_surface()
                        surface.save(filepath)
                    elif fmt == 'ply':
                        surface = image_data.extract_surface()
                        surface.save(filepath)
                    
                    exported_files.append(filename)
                    print(f"   ✓ {filename}")
                except Exception as e:
                    print(f"   ✗ {filename} - 失败: {str(e)}")
        
        # 分层导出
        if config.EXPORT_SEPARATE_LAYERS:
            print("\n2. 分层导出模型...")
            
            # 为每个地层创建单独的mesh
            for i in range(1, n_elements + 1):
                layer_name = element_names[i-1] if i-1 < len(element_names) else f"layer_{i}"
                print(f"\n   地层 {i}: {layer_name}")
                
                try:
                    # 使用threshold提取该地层
                    layer_mesh = image_data.threshold(
                        value=[i - 0.5, i + 0.5],
                        scalars="geological_formation"
                    )
                    
                    if layer_mesh.n_points == 0:
                        print(f"      ⚠ 跳过（无数据点）")
                        continue
                    
                    # 导出各种格式
                    for fmt in config.EXPORT_FORMATS:
                        try:
                            filename = f"layer_{i}_{layer_name}.{fmt}"
                            filepath = os.path.join(export_dir, filename)
                            
                            if fmt in ['vtk', 'vtu']:
                                layer_mesh.save(filepath)
                            elif fmt == 'stl':
                                surface = layer_mesh.extract_surface()
                                surface.save(filepath)
                            elif fmt == 'obj':
                                surface = layer_mesh.extract_surface()
                                surface.save(filepath)
                            elif fmt == 'ply':
                                surface = layer_mesh.extract_surface()
                                surface.save(filepath)
                            
                            exported_files.append(filename)
                            print(f"      ✓ {filename}")
                        except Exception as e:
                            print(f"      ✗ {filename} - 失败: {str(e)}")
                            
                except Exception as e:
                    print(f"      ✗ 地层提取失败: {str(e)}")
        
        # 导出摘要信息
        summary_file = os.path.join(export_dir, "export_summary.txt")
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("="*60 + "\n")
            f.write("地质模型导出摘要\n")
            f.write("="*60 + "\n\n")
            f.write(f"导出时间: {pd.Timestamp.now()}\n")
            f.write(f"模型名称: {config.PROJECT_NAME}\n")
            f.write(f"地层数量: {n_elements}\n")
            f.write(f"网格分辨率: {resolution}\n")
            f.write(f"模型范围: X[{extent[0]:.2f}, {extent[1]:.2f}], "
                   f"Y[{extent[2]:.2f}, {extent[3]:.2f}], "
                   f"Z[{extent[4]:.2f}, {extent[5]:.2f}]\n")
            f.write(f"Z轴缩放: {config.Z_SCALE}x\n\n")
            
            f.write("地层列表:\n")
            for i, name in enumerate(element_names, 1):
                f.write(f"  {i}. {name}\n")
            
            f.write(f"\n导出格式: {', '.join(config.EXPORT_FORMATS)}\n")
            f.write(f"导出文件数: {len(exported_files)}\n\n")
            
            f.write("导出文件列表:\n")
            for fname in sorted(exported_files):
                f.write(f"  - {fname}\n")
        
        print("\n" + "="*60)
        print(f"✓ 模型导出完成！")
        print(f"  导出文件数: {len(exported_files)}")
        print(f"  导出目录: {export_dir}")
        print(f"  摘要文件: {summary_file}")
        print("="*60)
        
    except Exception as e:
        print(f"\n✗ 模型导出失败: {str(e)}")
        import traceback
        traceback.print_exc()


def create_borehole_set(collar_df: pd.DataFrame, 
                       deviation_df: pd.DataFrame, 
                       lithology_df: pd.DataFrame) -> BoreholeSet:
    """
    将原始数据转换为subsurface的BoreholeSet对象
    
    Args:
        collar_df (pd.DataFrame): collar数据
        deviation_df (pd.DataFrame): deviation数据
        lithology_df (pd.DataFrame): lithology数据
    
    Returns:
        BoreholeSet: 包含完整钻孔信息的对象
    
    Notes:
        BoreholeSet对象整合了collar、survey和lithology信息，
        可用于后续的地质建模
    """
    print("\n正在创建钻孔数据对象...")
    
    # 1. 创建Collars对象
    print("  - 处理collar数据...")
    unstruc = ss.UnstructuredData.from_array(
        vertex=collar_df[["x", "y", "z"]].values,
        cells=SpecialCellCase.POINTS
    )
    points = ss.PointSet(data=unstruc)
    collars = Collars(
        ids=collar_df['hole_id'].tolist(),
        collar_loc=points
    )
    print(f"    完成: {len(collars.ids)} 个钻孔collar")
    
    # 2. 处理deviation数据为Survey对象
    print("  - 处理deviation数据...")
    # 为每个钻孔添加起始点(md=0)
    survey_records = []
    
    # 获取所有钻孔的最大深度（从lithology数据）
    max_depths = lithology_df.groupby('hole_id')['to_md'].max().to_dict()
    
    for hole_id in collar_df['hole_id'].unique():
        # 添加起始点 (md=0, inc=0, az=0)
        survey_records.append({
            'id': hole_id,
            'md': 0.0,
            'dip': 0.0,
            'azi': 0.0
        })
        
        # 添加该钻孔的实际deviation数据
        hole_dev = deviation_df[deviation_df['hole_id'] == hole_id]
        if len(hole_dev) > 0:
            for _, row in hole_dev.iterrows():
                survey_records.append({
                    'id': hole_id,
                    'md': float(row['md']),
                    'dip': float(row['inc']),
                    'azi': float(row['az'])
                })
        else:
            # 如果没有deviation数据，使用垂直钻孔假设
            max_depth = max_depths.get(hole_id, 100.0)
            survey_records.append({
                'id': hole_id,
                'md': max_depth,
                'dip': 0.0,  # 垂直
                'azi': 0.0
            })
    
    survey_df = pd.DataFrame(survey_records)
    survey_df = survey_df.sort_values(['id', 'md']).reset_index(drop=True)
    
    # 设置索引
    survey_df = survey_df.set_index('id')
    
    survey = Survey.from_df(survey_df)
    print(f"    完成: {len(survey_records)} 条survey记录")
    
    # 3. 处理lithology数据
    print("  - 处理lithology数据...")
    # 重命名列以匹配subsurface要求
    lith_df_renamed = lithology_df.rename(columns={
        'hole_id': 'id',
        'from_md': 'top',
        'to_md': 'base',
        'unit': 'component lith'
    })
    lith_df_renamed = lith_df_renamed.set_index('id')
    
    # 为每个岩性单元分配ID
    unique_units = lith_df_renamed['component lith'].unique()
    unit_to_id = {unit: idx + 1 for idx, unit in enumerate(unique_units)}
    lith_df_renamed['lith_ids'] = lith_df_renamed['component lith'].map(unit_to_id)
    
    survey.update_survey_with_lith(lith_df_renamed)
    print(f"    完成: {len(unique_units)} 种岩性单元")
    print(f"    岩性单元映射: {unit_to_id}")
    
    # 4. 创建BoreholeSet
    print("  - 创建BoreholeSet...")
    borehole_set = BoreholeSet(
        collars=collars,
        survey=survey,
        merge_option=MergeOptions.INTERSECT
    )
    print("    完成!")
    
    return borehole_set, unit_to_id


def visualize_boreholes(borehole_set: BoreholeSet, collars: Collars):
    """
    可视化钻孔轨迹和位置
    
    Args:
        borehole_set (BoreholeSet): 钻孔数据集
        collars (Collars): 钻孔collar对象
    """
    print("\n[可视化] 显示钻孔轨迹...")
    
    try:
        import matplotlib.pyplot as plt
        
        # 创建钻孔轨迹网格
        well_mesh = to_pyvista_line(
            line_set=borehole_set.combined_trajectory,
            active_scalar="lith_ids",
            radius=40
        )
        
        # 创建collar点网格
        collar_mesh = to_pyvista_points(collars.collar_loc)
        
        # 绘制
        pv_plot(
            [well_mesh, collar_mesh],
            image_2d=False,
            cmap="tab20c"
        )
        
    except Exception as e:
        print(f"  可视化时出错: {str(e)}")
        print("  跳过可视化步骤...")


# ==================== 主函数 ====================

def main():
    """
    主函数 - 执行完整的地质建模流程
    """
    print("="*60)
    print("GemPy 三维地质建模程序")
    print("="*60)
    
    # 步骤1: 读取数据
    print("\n[步骤1] 读取钻孔数据...")
    try:
        collar_df = load_collar_data(ModelConfig.DATA_DIR, ModelConfig.COLLAR_FILE)
        deviation_df = load_deviation_data(ModelConfig.DATA_DIR, ModelConfig.DEVIATION_FILE)
        lithology_df = load_lithology_data(ModelConfig.DATA_DIR, ModelConfig.LITHOLOGY_FILE)
    except Exception as e:
        print(f"错误: {str(e)}")
        return
    
    print("\n数据读取完成！")
    print(f"- Collar数据: {len(collar_df)} 个钻孔")
    print(f"- Deviation数据: {len(deviation_df)} 条记录")
    print(f"- Lithology数据: {len(lithology_df)} 条记录")
    
    # 步骤1.5: 处理地表数据
    print("\n[步骤1.5] 处理地表数据...")
    surface_df = None
    
    if ModelConfig.USE_SURFACE_TOPOGRAPHY:
        # 尝试读取地表数据文件
        surface_df = load_surface_data(ModelConfig.DATA_DIR, ModelConfig.SURFACE_DATA_FILE)
        
        if surface_df is None:
            # 如果没有地表文件，从collar数据创建
            print("  未找到地表数据文件，从collar数据推断...")
            surface_df = create_surface_points_from_collar(collar_df, ModelConfig)
        
        # 对地表进行网格插值
        if surface_df is not None:
            # 先计算初步范围用于插值
            temp_extent = [
                collar_df['x'].min(), collar_df['x'].max(),
                collar_df['y'].min(), collar_df['y'].max(),
                0, 0  # z暂时不用
            ]
            surface_df = interpolate_surface_grid(
                surface_df, 
                temp_extent, 
                ModelConfig.SURFACE_RESOLUTION
            )
            print(f"- Surface数据: {len(surface_df)} 个地表点")
    else:
        print("  地表地形功能已关闭（配置参数 USE_SURFACE_TOPOGRAPHY=False）")
    
    # 步骤2: 从钻孔数据创建结构元素（包含地表）
    print("\n[步骤2] 从钻孔数据创建结构元素...")
    try:
        elements, elements_dict, extent = create_structural_elements_from_data(
            collar_df, lithology_df, surface_df, ModelConfig
        )
        if surface_df is not None:
            print(f"\n结构元素创建成功！共 {len(elements)} 个单元（含地表）")
        else:
            print(f"\n结构元素创建成功！共 {len(elements)} 个地层单元")
    except Exception as e:
        print(f"错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return
    
    # 步骤3: 创建GemPy模型
    print("\n[步骤3] 创建GemPy地质模型...")
    try:
        geo_model = create_gempy_model(elements, extent, ModelConfig)
        print("GemPy模型创建成功！")
    except Exception as e:
        print(f"错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return
    
    # 步骤4: 计算模型并可视化
    print("\n[步骤4] 计算模型并可视化...")
    try:
        compute_and_visualize_model(geo_model, ModelConfig)
    except Exception as e:
        print(f"错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return
    
    # 步骤5: 导出模型
    try:
        export_model(geo_model, ModelConfig)
    except Exception as e:
        print(f"导出模型时出错: {str(e)}")
        import traceback
        traceback.print_exc()
        # 导出失败不影响主流程，继续执行
    
    print("\n" + "="*60)
    print("建模完成！")
    print("="*60)
    
    return geo_model


if __name__ == "__main__":
    main()


# ==================== 使用说明 ====================
"""
使用说明：
---------

1. 环境准备：
   确保安装以下Python包：
   - gempy >= 3.0.0
   - gempy-viewer >= 2025.1.0
   - pandas >= 2.0.0
   - numpy >= 1.24.0
   - subsurface >= 0.3.0
   - pyvista >= 0.43.0

   安装命令：
   pip install gempy[base] gempy-viewer pandas numpy subsurface pyvista scipy

2. 数据准备：
   将以下CSV文件放在 src_Implicit/data/ 目录下：
   - collars.csv: 钻孔孔口坐标 (hole_id, x, y, z)
   - deviation.csv: 钻孔偏斜数据 (hole_id, md, inc, az) [当前版本暂不使用]
   - lithology.csv: 岩性数据 (hole_id, from_md, to_md, unit)
   - surface_points.csv: 地表高程数据 (x, y, z) [可选，如果没有会自动从collar推断]

3. 参数调整：
   在 ModelConfig 类中可以调整以下参数：
   
   模型参数：
   - GRID_RESOLUTION: 网格分辨率，影响计算速度和精度
   - INTERPOLATION_RANGE: 插值范围
   - C_O: 协方差参数
   - NUMBER_OCTREE_LEVELS: 八叉树层级
   
   可视化参数：
   - SHOW_3D_PLOT: 是否显示3D图
   - SHOW_2D_PLOT: 是否显示2D图
   - Z_SCALE: Z轴缩放比例（垂直夸大系数）
   - SHOW_BOREHOLE_POINTS: 是否显示钻孔数据点
   - SHOW_LAYER_VOLUME: 是否显示完整地层体积模型
   - LAYER_OPACITY: 地层透明度 (0-1)
   
   地表参数：
   - USE_SURFACE_TOPOGRAPHY: 是否使用地表地形（True=使用真实地形）
   - SURFACE_DATA_FILE: 地表数据文件名
   - SURFACE_RESOLUTION: 地表网格分辨率
   - SURFACE_ELEVATION: 固定地表高程（None则从collar推断）
   
   导出参数：
   - EXPORT_MODEL: 是否导出模型
   - EXPORT_FORMATS: 导出格式列表 ['vtk', 'vtu', 'stl', 'obj', 'ply']
   - EXPORT_SEPARATE_LAYERS: 是否分层导出
   - EXPORT_COMBINED_MODEL: 是否导出合并模型

4. 运行程序：
   python gempy_borehole_modeling.py

5. 输出结果：
   - 控制台会显示建模过程和统计信息
   - 自动弹出2D地质剖面图
   - 交互式3D可视化窗口
   - 模型文件导出到 output/exported_models/ 目录
   - 支持的导出格式：
     * VTK/VTU: 可用ParaView打开
     * STL: 可用于3D打印
     * OBJ: 可用Blender等软件打开
     * PLY: 点云格式

6. 当前实现特点：
   - 假设所有钻孔为垂直钻孔（不使用deviation数据）
   - 支持真实地表地形（从文件读取或collar推断）
   - 自动为每个地层生成水平方位（法向量指向上方）
   - 支持多个地层单元的建模
   - 完整3D体积模型可视化和导出
   - 自动计算模型范围

7. 地表数据说明：
   - 如果有真实DEM数据，创建surface_points.csv文件
   - 如果没有地表文件，程序会自动使用collar点高程作为地表
   - 地表会被自动插值成规则网格（分辨率可调）
   - 模型顶部会跟随真实地表起伏，而不是平面

8. 后续改进方向：
   - 集成deviation数据支持倾斜钻孔
   - 优化方位数据生成算法
   - 添加更多可视化选项
   - 支持断层建模
"""
