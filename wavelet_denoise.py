#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
小波阈值去噪法处理FITS文件
使用小波变换去除天文图像中的噪声
"""

import numpy as np
from astropy.io import fits
import pywt
import matplotlib.pyplot as plt
from scipy import ndimage
import os
import argparse

def wavelet_denoise(image, wavelet='bior4.4', sigma=None, mode='soft', method='adaptive',
                   levels=6, threshold_factor=0.1):
    """
    使用小波阈值去噪法去除图像噪声，针对小面积锐利噪点优化

    参数:
    image: 输入图像数组
    wavelet: 小波基函数，默认'bior4.4'（双正交小波，更适合锐利特征）
    sigma: 噪声标准差，如果为None则自动估计
    mode: 阈值模式，'soft'或'hard'
    method: 阈值选择方法，'adaptive'、'bayes'或'sure'
    levels: 分解级数，默认6（更多级数捕获更细小的噪点）
    threshold_factor: 阈值因子，用于调整阈值敏感度

    返回:
    去噪后的图像
    """

    # 如果没有提供噪声标准差，则估计噪声水平
    if sigma is None:
        # 使用最高频子带估计噪声标准差
        coeffs = pywt.wavedec2(image, wavelet, level=1)
        # 使用更保守的噪声估计，针对锐利噪点
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745
        print(f"估计的噪声标准差: {sigma:.4f}")

    # 多级小波分解，使用更多级数捕获细小噪点
    coeffs = pywt.wavedec2(image, wavelet, level=levels)

    # 自适应阈值计算，针对不同频率子带使用不同阈值
    coeffs_thresh = list(coeffs)
    coeffs_thresh[0] = coeffs[0]  # 保持低频部分不变

    print(f"使用{levels}级小波分解，小波基函数: {wavelet}")

    for i in range(1, len(coeffs)):
        # 为每个分解级别计算自适应阈值
        level_detail = coeffs[i]

        if method == 'adaptive':
            # 自适应阈值：对高频子带使用更低的阈值以保留锐利特征
            level_factor = threshold_factor * (0.5 ** (i-1))  # 高频级别使用更小的阈值

            # 为每个方向（水平、垂直、对角）计算独立阈值
            thresh_details = []
            for j, detail in enumerate(level_detail):
                # 计算该子带的局部噪声水平
                local_sigma = np.median(np.abs(detail)) / 0.6745

                # 针对锐利噪点的阈值策略
                if j == 2:  # 对角分量，通常包含更多噪声
                    threshold = local_sigma * level_factor * 2.0
                else:  # 水平和垂直分量，保留更多边缘信息
                    threshold = local_sigma * level_factor * 1.5

                # 应用阈值
                thresh_detail = pywt.threshold(detail, threshold, mode=mode)
                thresh_details.append(thresh_detail)

            coeffs_thresh[i] = tuple(thresh_details)
            print(f"级别 {i}: 自适应阈值 {[f'{local_sigma * level_factor * (2.0 if j==2 else 1.5):.4f}' for j in range(3)]}")

        elif method == 'bayes':
            # BayesShrink阈值，针对锐利噪点调整
            threshold = sigma * threshold_factor * np.sqrt(2 * np.log(image.size))
            coeffs_thresh[i] = tuple([
                pywt.threshold(detail, threshold, mode=mode)
                for detail in level_detail
            ])
            print(f"级别 {i}: BayesShrink阈值 {threshold:.4f}")

        else:  # SURE或其他方法
            threshold = sigma * threshold_factor * np.sqrt(2 * np.log(image.size))
            coeffs_thresh[i] = tuple([
                pywt.threshold(detail, threshold, mode=mode)
                for detail in level_detail
            ])
            print(f"级别 {i}: 标准阈值 {threshold:.4f}")

    # 重构图像
    denoised = pywt.waverec2(coeffs_thresh, wavelet)

    # 确保重构后的图像尺寸与原图像一致
    if denoised.shape != image.shape:
        # 裁剪到原始尺寸
        denoised = denoised[:image.shape[0], :image.shape[1]]

    return denoised

def extract_noise(original, denoised):
    """
    提取噪声部分
    """
    noise = original - denoised
    return noise

def process_fits_file(input_file, output_file, noise_file=None, wavelet='bior4.4',
                     sigma=None, mode='soft', method='adaptive', levels=6, threshold_factor=0.1):
    """
    处理FITS文件
    
    参数:
    input_file: 输入FITS文件路径
    output_file: 输出去噪FITS文件路径
    noise_file: 输出噪声FITS文件路径（可选）
    其他参数同wavelet_denoise函数
    """
    
    print(f"正在读取FITS文件: {input_file}")
    
    # 读取FITS文件
    with fits.open(input_file) as hdul:
        header = hdul[0].header
        image_data = hdul[0].data.astype(np.float64)
        
        print(f"图像尺寸: {image_data.shape}")
        print(f"数据类型: {image_data.dtype}")
        print(f"数据范围: [{np.min(image_data):.2f}, {np.max(image_data):.2f}]")
    
    # 处理NaN值
    if np.any(np.isnan(image_data)):
        print("检测到NaN值，将其替换为中位数")
        median_val = np.nanmedian(image_data)
        image_data = np.nan_to_num(image_data, nan=median_val)
    
    # 小波去噪
    print("开始小波去噪处理...")
    denoised_image = wavelet_denoise(image_data, wavelet=wavelet, sigma=sigma,
                                   mode=mode, method=method, levels=levels,
                                   threshold_factor=threshold_factor)
    
    # 提取噪声
    noise_image = extract_noise(image_data, denoised_image)
    
    # 保存去噪后的图像
    print(f"保存去噪图像到: {output_file}")
    fits.writeto(output_file, denoised_image, header=header, overwrite=True)
    
    # 保存噪声图像（如果指定了路径）
    if noise_file:
        print(f"保存噪声图像到: {noise_file}")
        fits.writeto(noise_file, noise_image, header=header, overwrite=True)
    
    # 显示统计信息
    print("\n处理结果统计:")
    print(f"原始图像 - 均值: {np.mean(image_data):.4f}, 标准差: {np.std(image_data):.4f}")
    print(f"去噪图像 - 均值: {np.mean(denoised_image):.4f}, 标准差: {np.std(denoised_image):.4f}")
    print(f"噪声图像 - 均值: {np.mean(noise_image):.4f}, 标准差: {np.std(noise_image):.4f}")
    
    return denoised_image, noise_image

def plot_comparison(original, denoised, noise, save_path=None):
    """
    绘制对比图
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 原始图像
    im1 = axes[0].imshow(original, cmap='gray', origin='lower')
    axes[0].set_title('原始图像')
    axes[0].axis('off')
    plt.colorbar(im1, ax=axes[0])
    
    # 去噪图像
    im2 = axes[1].imshow(denoised, cmap='gray', origin='lower')
    axes[1].set_title('去噪图像')
    axes[1].axis('off')
    plt.colorbar(im2, ax=axes[1])
    
    # 噪声图像
    im3 = axes[2].imshow(noise, cmap='gray', origin='lower')
    axes[2].set_title('提取的噪声')
    axes[2].axis('off')
    plt.colorbar(im3, ax=axes[2])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"对比图保存到: {save_path}")
    
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='使用小波阈值去噪法处理FITS文件')
    parser.add_argument('input_file', help='输入FITS文件路径')
    parser.add_argument('-o', '--output', help='输出去噪FITS文件路径', 
                       default='denoised_output.fits')
    parser.add_argument('-n', '--noise', help='输出噪声FITS文件路径', 
                       default='noise_output.fits')
    parser.add_argument('-w', '--wavelet', help='小波基函数', default='bior4.4')
    parser.add_argument('-s', '--sigma', type=float, help='噪声标准差（自动估计如果未指定）')
    parser.add_argument('-m', '--mode', choices=['soft', 'hard'],
                       help='阈值模式', default='soft')
    parser.add_argument('--method', choices=['adaptive', 'bayes', 'sure'],
                       help='阈值选择方法', default='adaptive')
    parser.add_argument('-l', '--levels', type=int, help='小波分解级数', default=6)
    parser.add_argument('-t', '--threshold-factor', type=float,
                       help='阈值因子（越小越敏感）', default=0.1)
    parser.add_argument('--plot', action='store_true', help='显示对比图')
    parser.add_argument('--save-plot', help='保存对比图路径')
    
    args = parser.parse_args()
    
    # 检查输入文件是否存在
    if not os.path.exists(args.input_file):
        print(f"错误: 输入文件 {args.input_file} 不存在")
        return
    
    try:
        # 处理FITS文件
        denoised, noise = process_fits_file(
            args.input_file, args.output, args.noise,
            wavelet=args.wavelet, sigma=args.sigma,
            mode=args.mode, method=args.method,
            levels=args.levels, threshold_factor=args.threshold_factor
        )
        
        # 如果需要显示或保存对比图
        if args.plot or args.save_plot:
            with fits.open(args.input_file) as hdul:
                original = hdul[0].data.astype(np.float64)
            plot_comparison(original, denoised, noise, args.save_plot)
        
        print("\n处理完成!")
        
    except Exception as e:
        print(f"处理过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # 如果直接运行脚本，处理当前目录下的FITS文件
    fits_files = [f for f in os.listdir('.') if f.endswith('.fit') or f.endswith('.fits')]
    
    if fits_files:
        input_file = fits_files[0]  # 使用第一个找到的FITS文件
        print(f"找到FITS文件: {input_file}")
        
        # 生成输出文件名
        base_name = os.path.splitext(input_file)[0]
        output_file = f"{base_name}_denoised.fits"
        noise_file = f"{base_name}_noise.fits"
        
        try:
            # 针对锐利噪点的优化参数
            denoised, noise = process_fits_file(
                input_file, output_file, noise_file,
                wavelet='bior4.4',  # 双正交小波，更适合锐利特征
                levels=8,           # 更多分解级数捕获细小噪点
                threshold_factor=0.03,  # 更低的阈值因子，更敏感
                method='adaptive',  # 自适应阈值
                mode='soft'        # 软阈值保持平滑性
            )
            print(f"\n处理完成!")
            print(f"去噪图像保存为: {output_file}")
            print(f"噪声图像保存为: {noise_file}")

        except Exception as e:
            print(f"处理过程中出现错误: {str(e)}")
            import traceback
            traceback.print_exc()
    else:
        print("当前目录下没有找到FITS文件")
        main()  # 使用命令行参数模式
