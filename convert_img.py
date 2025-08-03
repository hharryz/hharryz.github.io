import os
import re
from pathlib import Path
from PIL import Image
import argparse

def convert_image_to_webp(input_path, output_path):
    """
    将图片转换为WebP格式（无损压缩）
    """
    try:
        with Image.open(input_path) as img:
            # 如果是PNG且有透明通道，保持RGBA模式
            if img.format == 'PNG' and img.mode in ('RGBA', 'LA'):
                img.save(output_path, 'WEBP', lossless=True, quality=100)
            else:
                # 转换为RGB模式（适用于JPG等格式）
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                img.save(output_path, 'WEBP', lossless=True, quality=100)
        return True
    except Exception as e:
        print(f"转换失败 {input_path}: {e}")
        return False

def scan_and_convert_images(folder_path, keep_original=False):
    """
    扫描img文件夹并转换图片
    """
    img_folder = Path(folder_path) / 'img'
    if not img_folder.exists():
        print(f"未找到img文件夹: {img_folder}")
        return []

    converted_files = []
    supported_formats = {'.png', '.jpg', '.jpeg'}

    for img_file in img_folder.iterdir():
        if img_file.suffix.lower() in supported_formats:
            webp_file = img_file.with_suffix('.webp')

            print(f"正在转换: {img_file.name} -> {webp_file.name}")

            if convert_image_to_webp(img_file, webp_file):
                converted_files.append({
                    'original': img_file.name,
                    'converted': webp_file.name,
                    'original_suffix': img_file.suffix.lower()
                })
                # 根据选项决定是否删除原文件
                if not keep_original:
                    try:
                        img_file.unlink()
                        print(f"已删除原文件: {img_file.name}")
                    except Exception as e:
                        print(f"删除原文件失败 {img_file.name}: {e}")
                else:
                    print(f"已保留原文件: {img_file.name}")

    return converted_files

def update_markdown_references(folder_path, converted_files):
    """
    更新Markdown文件中的图片引用
    """
    folder = Path(folder_path)
    md_files = list(folder.glob('*.md'))

    if not md_files:
        print("未找到Markdown文件")
        return

    # 创建替换映射
    replacement_map = {}
    for file_info in converted_files:
        original_name = file_info['original']
        converted_name = file_info['converted']
        # 移除扩展名的原始名称
        base_name = Path(original_name).stem
        original_suffix = file_info['original_suffix']

        # 创建多种可能的引用模式
        patterns = [
            f"img/{original_name}",
            f"./img/{original_name}",
            f"img/{base_name}{original_suffix}",
            f"./img/{base_name}{original_suffix}"
        ]

        for pattern in patterns:
            replacement_map[pattern] = f"img/{converted_name}"

    # 更新每个Markdown文件
    for md_file in md_files:
        try:
            with open(md_file, 'r', encoding='utf-8') as f:
                content = f.read()

            original_content = content
            updated = False

            # 替换图片引用
            for old_ref, new_ref in replacement_map.items():
                # 匹配Markdown图片语法: ![alt](path) 和 <img src="path">
                patterns = [
                    (rf'(\!\[[^\]]*\]\()\s*{re.escape(old_ref)}\s*(\))', rf'\1{new_ref}\2'),
                    (rf'(<img[^>]+src=["\'\'"])\s*{re.escape(old_ref)}\s*(["\'\'"][^>]*>)', rf'\1{new_ref}\2')
                ]

                for pattern, replacement in patterns:
                    if re.search(pattern, content):
                        content = re.sub(pattern, replacement, content)
                        updated = True

            # 如果有更新，写回文件
            if updated:
                with open(md_file, 'w', encoding='utf-8') as f:
                    f.write(content)
                print(f"已更新Markdown文件: {md_file.name}")

        except Exception as e:
            print(f"更新Markdown文件失败 {md_file}: {e}")

def main():
    parser = argparse.ArgumentParser(description='将PNG/JPG图片转换为WebP格式并更新Markdown引用')
    parser.add_argument('--folder', required=True, help='包含img文件夹和Markdown文件的目标文件夹路径')
    parser.add_argument('--keep-original', action='store_true', help='保留原始图片文件，不删除')

    args = parser.parse_args()
    folder_path = Path(args.folder)

    if not folder_path.exists():
        print(f"错误: 文件夹不存在 - {folder_path}")
        return

    if not folder_path.is_dir():
        print(f"错误: 路径不是文件夹 - {folder_path}")
        return

    print(f"开始处理文件夹: {folder_path}")
    print("=" * 50)

    # 步骤1: 转换图片
    print("1. 扫描并转换图片...")
    converted_files = scan_and_convert_images(folder_path, args.keep_original)

    if not converted_files:
        print("未找到需要转换的图片文件")
        return

    print(f"\n成功转换 {len(converted_files)} 个文件:")
    for file_info in converted_files:
        print(f"  {file_info['original']} -> {file_info['converted']}")

    # 步骤2: 更新Markdown文件
    print("\n2. 更新Markdown文件中的图片引用...")
    update_markdown_references(folder_path, converted_files)

    print("\n" + "=" * 50)
    print("转换完成!")

if __name__ == "__main__":
    main()
