import pandas as pd

def process_thickness_to_coordinates():
    # 读取地层统计数据
    layer_data_path = './data/real_data/地层统计_标准分段_合并结果.xlsx'
    layer_data = pd.read_excel(layer_data_path)
    # print(layer_data)
    # 读取钻孔位置数据
    borehole_data_path = './data/real_data/钻孔位置统计.xlsx'
    borehole_data = pd.read_excel(borehole_data_path)
    # print(borehole_data)
    # 合并数据，计算地层底板坐标
    results = []
    for _, borehole in borehole_data.iterrows():
        borehole_name = borehole['钻孔名称']
        x, y, z_top = borehole['x'], borehole['y'], borehole['z']

        # 筛选对应钻孔的地层数据
        borehole_layers = layer_data[layer_data['钻孔名称'] == borehole_name]

        z_current = z_top

        for _, layer in borehole_layers.iterrows():
            layer_name = layer['地层名称']
            thickness = layer['厚度']
            z_bottom = z_current - thickness

            # 保存地层底板坐标
            results.append({
                '地层名称': layer_name,
                'x': x,
                'y': y,
                'z': z_bottom
            })

            z_current = z_bottom

    # 添加地表层数据到地层坐标
    surface_data = borehole_data[['钻孔名称', 'x', 'y', 'z']].copy()
    surface_data.rename(columns={'钻孔名称': '地层名称'}, inplace=True)
    surface_data['地层名称'] = '地表层'

    # 合并地表层和地层数据
    results_df = pd.DataFrame(results)
    combined_df = pd.concat([surface_data, results_df], ignore_index=True)

    # 按地层名称排序
    combined_df.sort_values(by=['地层名称'], inplace=True)

    # 写出地层坐标数据
    output_path = './data/real_data/地层坐标.xlsx'
    combined_df.to_excel(output_path, index=False)

process_thickness_to_coordinates()