#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
专门针对锐利噪点的配置脚本
使用极其敏感的参数来捕获最小的噪点
"""

import os
from wavelet_denoise import process_fits_file

def process_sharp_noise(input_file, output_suffix="_sharp_denoised"):
    """
    使用专门针对锐利噪点的参数处理FITS文件
    """
    
    # 生成输出文件名
    base_name = os.path.splitext(input_file)[0]
    output_file = f"{base_name}{output_suffix}.fits"
    noise_file = f"{base_name}_sharp_noise.fits"
    
    print("=== 锐利噪点专用处理模式 ===")
    print("参数配置:")
    print("- 小波基函数: bior6.8 (高阶双正交小波，更好的频率分辨率)")
    print("- 分解级数: 10 (极深分解，捕获最细微的噪点)")
    print("- 阈值因子: 0.01 (极低阈值，最大敏感度)")
    print("- 阈值模式: hard (硬阈值，更锐利的去噪效果)")
    print("- 方法: adaptive (自适应阈值)")
    print()
    
    try:
        denoised, noise = process_fits_file(
            input_file, output_file, noise_file,
            wavelet='bior6.8',      # 高阶双正交小波，更好的频率分辨率
            levels=10,              # 极深分解级数
            threshold_factor=0.01,  # 极低阈值因子，最大敏感度
            method='adaptive',      # 自适应阈值
            mode='hard'            # 硬阈值，更锐利的去噪效果
        )
        
        print(f"\n🎯 锐利噪点处理完成!")
        print(f"📁 去噪图像: {output_file}")
        print(f"📁 噪声图像: {noise_file}")
        
        return denoised, noise
        
    except Exception as e:
        print(f"❌ 处理过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None

def process_ultra_sharp_noise(input_file, output_suffix="_ultra_sharp_denoised"):
    """
    使用极端参数处理最细微的锐利噪点
    """
    
    # 生成输出文件名
    base_name = os.path.splitext(input_file)[0]
    output_file = f"{base_name}{output_suffix}.fits"
    noise_file = f"{base_name}_ultra_sharp_noise.fits"
    
    print("=== 超锐利噪点专用处理模式 ===")
    print("参数配置:")
    print("- 小波基函数: coif5 (Coiflets小波，优秀的时频局域化)")
    print("- 分解级数: 12 (超深分解)")
    print("- 阈值因子: 0.005 (超低阈值)")
    print("- 阈值模式: hard (硬阈值)")
    print("- 方法: adaptive (自适应阈值)")
    print()
    
    try:
        denoised, noise = process_fits_file(
            input_file, output_file, noise_file,
            wavelet='coif5',        # Coiflets小波，优秀的时频局域化
            levels=12,              # 超深分解级数
            threshold_factor=0.005, # 超低阈值因子
            method='adaptive',      # 自适应阈值
            mode='hard'            # 硬阈值
        )
        
        print(f"\n🎯 超锐利噪点处理完成!")
        print(f"📁 去噪图像: {output_file}")
        print(f"📁 噪声图像: {noise_file}")
        
        return denoised, noise
        
    except Exception as e:
        print(f"❌ 处理过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None

def main():
    # 查找FITS文件
    fits_files = [f for f in os.listdir('.') if f.endswith('.fit') or f.endswith('.fits')]
    
    if not fits_files:
        print("❌ 当前目录下没有找到FITS文件")
        return
    
    input_file = fits_files[0]  # 使用第一个找到的FITS文件
    print(f"🔍 找到FITS文件: {input_file}")
    print()
    
    # 处理锐利噪点
    print("1️⃣ 开始锐利噪点处理...")
    process_sharp_noise(input_file)
    print()
    
    # 处理超锐利噪点
    print("2️⃣ 开始超锐利噪点处理...")
    process_ultra_sharp_noise(input_file)
    print()
    
    print("✅ 所有处理完成!")

if __name__ == "__main__":
    main()
