#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
噪点检测结果总结
对比不同方法的检测效果
"""

import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
import os

def analyze_fits_file(filename):
    """分析FITS文件的统计信息"""
    
    if not os.path.exists(filename):
        return None
    
    with fits.open(filename) as hdul:
        data = hdul[0].data.astype(np.float64)
        
        stats = {
            'filename': filename,
            'shape': data.shape,
            'mean': np.mean(data),
            'std': np.std(data),
            'min': np.min(data),
            'max': np.max(data),
            'non_zero_pixels': np.sum(data != 0),
            'total_pixels': data.size
        }
        
        return stats

def compare_noise_detection_results():
    """对比不同噪点检测方法的结果"""
    
    print("🔍 噪点检测结果对比分析")
    print("=" * 60)
    
    # 查找原始文件
    fits_files = [f for f in os.listdir('.') if f.endswith('.fit')]
    if not fits_files:
        print("❌ 未找到原始FITS文件")
        return
    
    original_file = fits_files[0]
    base_name = os.path.splitext(original_file)[0]
    
    print(f"📁 原始文件: {original_file}")
    
    # 分析原始图像
    original_stats = analyze_fits_file(original_file)
    if original_stats:
        print(f"\n📊 原始图像统计:")
        print(f"  尺寸: {original_stats['shape']}")
        print(f"  总像素数: {original_stats['total_pixels']:,}")
        print(f"  均值: {original_stats['mean']:.2f}")
        print(f"  标准差: {original_stats['std']:.2f}")
        print(f"  数据范围: [{original_stats['min']:.0f}, {original_stats['max']:.0f}]")
    
    # 定义要分析的文件类型
    analysis_files = [
        # 小波去噪结果
        (f"{base_name}_denoised.fits", "标准小波去噪", "去噪图像"),
        (f"{base_name}_noise.fits", "标准小波噪声", "噪声提取"),
        (f"{base_name}_sharp_denoised.fits", "锐利小波去噪", "去噪图像"),
        (f"{base_name}_sharp_noise.fits", "锐利小波噪声", "噪声提取"),
        (f"{base_name}_ultra_sharp_denoised.fits", "超锐利小波去噪", "去噪图像"),
        (f"{base_name}_ultra_sharp_noise.fits", "超锐利小波噪声", "噪声提取"),
        
        # 单像素检测结果
        (f"{base_name}_simple_repaired.fits", "单像素修复", "去噪图像"),
        (f"{base_name}_simple_noise.fits", "单像素噪声", "噪声提取"),
        (f"{base_name}_hot_pixels_simple.fits", "热像素检测", "噪声提取"),
        (f"{base_name}_cold_pixels_simple.fits", "冷像素检测", "噪声提取"),
    ]
    
    print(f"\n🔬 各种方法检测结果:")
    print("-" * 80)
    print(f"{'方法':<20} {'类型':<12} {'噪点数量':<12} {'噪点占比':<12} {'噪声强度':<12}")
    print("-" * 80)
    
    results = []
    
    for filename, method_name, file_type in analysis_files:
        stats = analyze_fits_file(filename)
        if stats:
            if file_type == "噪声提取":
                noise_count = stats['non_zero_pixels']
                noise_ratio = (noise_count / stats['total_pixels']) * 100
                noise_intensity = stats['std'] if noise_count > 0 else 0
                
                print(f"{method_name:<20} {file_type:<12} {noise_count:<12,} {noise_ratio:<12.6f}% {noise_intensity:<12.2f}")
                
                results.append({
                    'method': method_name,
                    'type': file_type,
                    'noise_count': noise_count,
                    'noise_ratio': noise_ratio,
                    'noise_intensity': noise_intensity
                })
            else:
                # 对于去噪图像，计算与原始图像的差异
                if original_stats:
                    mean_diff = abs(stats['mean'] - original_stats['mean'])
                    std_diff = abs(stats['std'] - original_stats['std'])
                    
                    print(f"{method_name:<20} {file_type:<12} {'N/A':<12} {'N/A':<12} {std_diff:<12.2f}")
    
    # 分析单像素检测的详细结果
    print(f"\n🎯 单像素噪点检测详细分析:")
    print("-" * 50)
    
    # 热像素分析
    hot_stats = analyze_fits_file(f"{base_name}_hot_pixels_simple.fits")
    cold_stats = analyze_fits_file(f"{base_name}_cold_pixels_simple.fits")
    
    if hot_stats and cold_stats:
        hot_count = hot_stats['non_zero_pixels']
        cold_count = cold_stats['non_zero_pixels']
        total_single_pixels = hot_count + cold_count
        
        print(f"热像素数量: {hot_count:,}")
        print(f"冷像素数量: {cold_count:,}")
        print(f"单像素噪点总数: {total_single_pixels:,}")
        print(f"单像素噪点占比: {(total_single_pixels / original_stats['total_pixels']) * 100:.6f}%")
        
        if hot_count > 0:
            with fits.open(f"{base_name}_hot_pixels_simple.fits") as hdul:
                hot_data = hdul[0].data
                hot_values = hot_data[hot_data > 0]
                print(f"热像素值范围: [{np.min(hot_values):.0f}, {np.max(hot_values):.0f}]")
                print(f"热像素平均值: {np.mean(hot_values):.2f}")
        
        if cold_count > 0:
            with fits.open(f"{base_name}_cold_pixels_simple.fits") as hdul:
                cold_data = hdul[0].data
                cold_values = cold_data[cold_data > 0]
                print(f"冷像素值范围: [{np.min(cold_values):.0f}, {np.max(cold_values):.0f}]")
                print(f"冷像素平均值: {np.mean(cold_values):.2f}")
    
    # 方法对比总结
    print(f"\n📈 方法对比总结:")
    print("-" * 50)
    print("1. 小波去噪方法:")
    print("   - 标准小波: 适合一般噪声去除")
    print("   - 锐利小波: 针对小面积锐利噪点优化")
    print("   - 超锐利小波: 极端敏感度，捕获最细微噪点")
    print()
    print("2. 单像素检测方法:")
    print("   - 热像素检测: 检测异常亮的单像素")
    print("   - 冷像素检测: 检测异常暗的单像素")
    print("   - 修复效果: 保持图像整体特征的同时去除孤立噪点")
    
    # 推荐使用场景
    print(f"\n💡 推荐使用场景:")
    print("-" * 50)
    print("• 天文图像处理: 使用单像素检测去除热像素和冷像素")
    print("• 一般去噪: 使用标准小波去噪")
    print("• 精细处理: 使用锐利小波去噪")
    print("• 极端情况: 使用超锐利小波去噪")
    
    return results

def create_detection_report():
    """创建检测报告"""
    
    print("\n📋 生成检测报告...")
    
    results = compare_noise_detection_results()
    
    # 生成报告文件
    report_filename = "noise_detection_report.txt"
    
    with open(report_filename, 'w', encoding='utf-8') as f:
        f.write("噪点检测分析报告\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("处理的文件:\n")
        fits_files = [f for f in os.listdir('.') if f.endswith('.fit')]
        if fits_files:
            f.write(f"原始文件: {fits_files[0]}\n\n")
        
        f.write("生成的处理结果文件:\n")
        result_files = [f for f in os.listdir('.') if f.endswith('.fits')]
        for file in sorted(result_files):
            f.write(f"- {file}\n")
        
        f.write(f"\n检测到的文件总数: {len(result_files)}\n")
        
        f.write("\n处理方法说明:\n")
        f.write("1. 小波阈值去噪: 使用小波变换去除噪声\n")
        f.write("2. 单像素检测: 专门检测和修复单个像素的异常值\n")
        f.write("3. 热像素检测: 检测异常亮的像素\n")
        f.write("4. 冷像素检测: 检测异常暗的像素\n")
    
    print(f"📄 检测报告已保存到: {report_filename}")

def main():
    try:
        create_detection_report()
        print(f"\n✅ 分析完成!")
        
    except Exception as e:
        print(f"❌ 分析过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
