# metadata_parser.py

import os
import argparse
import pandas as pd
from pathlib import Path
from typing import List, Optional, Dict, Any

def load_metadata(metadata_dir: Path) -> pd.DataFrame:
    """
    从 metadata_dir 目录下加载所有 .xlsx 元数据文件，并合并为一个 DataFrame。
    每个 .xlsx 文件对应一个视频片段（scene），文件名作为 video_name 加入数据。
    """
    all_dfs = []
    for xlsx_path in metadata_dir.rglob("*.xlsx"):
        try:
            df = pd.read_excel(xlsx_path, engine='openpyxl')
            # 添加 video_name 字段（不含扩展名）
            video_name = xlsx_path.stem
            df["video_name"] = video_name
            all_dfs.append(df)
        except Exception as e:
            print(f"Warning: Failed to load {xlsx_path}: {e}")
    
    if not all_dfs:
        raise ValueError(f"No .xlsx files found in {metadata_dir}")
    
    combined_df = pd.concat(all_dfs, ignore_index=True)
    return combined_df

def filter_by_height(df: pd.DataFrame, min_height: Optional[float] = None, max_height: Optional[float] = None) -> pd.DataFrame:
    """按采集高度（acquisition_height）筛选"""
    if min_height is not None:
        df = df[df["acquisition_height"] >= min_height]
    if max_height is not None:
        df = df[df["acquisition_height"] <= max_height]
    return df

def filter_by_weather(df: pd.DataFrame, weathers: List[str]) -> pd.DataFrame:
    """按天气（acquisition_weather）筛选，支持多值匹配（模糊匹配）"""
    if not weathers:
        return df
    mask = df["acquisition_weather"].str.contains('|'.join(weathers), case=False, na=False)
    return df[mask]

def filter_by_scene(df: pd.DataFrame, scenes: List[str]) -> pd.DataFrame:
    """按场景类型（acquisition_scene）筛选，支持多值匹配（模糊匹配）"""
    if not scenes:
        return df
    mask = df["acquisition_scene"].str.contains('|'.join(scenes), case=False, na=False)
    return df[mask]

def filter_by_data_source(df: pd.DataFrame, sources: List[str]) -> pd.DataFrame:
    """按数据源类型（data_source_type）筛选，如 ['可见光', '红外']"""
    if not sources:
        return df
    mask = df["data_source_type"].isin(sources)
    return df[mask]

def filter_metadata(
    df: pd.DataFrame,
    min_height: Optional[float] = None,
    max_height: Optional[float] = None,
    weathers: Optional[List[str]] = None,
    scenes: Optional[List[str]] = None,
    data_sources: Optional[List[str]] = None,
) -> pd.DataFrame:
    """组合筛选函数"""
    df = filter_by_height(df, min_height, max_height)
    if weathers:
        df = filter_by_weather(df, weathers)
    if scenes:
        df = filter_by_scene(df, scenes)
    if data_sources:
        df = filter_by_data_source(df, data_sources)
    return df

def main():
    parser = argparse.ArgumentParser(
        description="Parse NEODrone metadata (.xlsx) and filter subsets by conditions."
    )
    parser.add_argument("--metadata_dir", type=str, required=True, help="Path to metadata/ directory containing .xlsx files")
    parser.add_argument("--output_csv", type=str, default="filtered_metadata.csv", help="Output CSV file path")
    
    # 筛选条件
    parser.add_argument("--min_height", type=float, help="Minimum acquisition height (meters)")
    parser.add_argument("--max_height", type=float, help="Maximum acquisition height (meters)")
    parser.add_argument("--weather", type=str, nargs="*", help="Weather conditions to include (e.g., 晴天 雾 小雨)")
    parser.add_argument("--scene", type=str, nargs="*", help="Scene types to include (e.g., 城市 乡村 交通)")
    parser.add_argument("--source", type=str, nargs="*", help="Data source types (e.g., 可见光 红外)")

    parser.add_argument("--list_unique", action="store_true", help="List unique values of key fields and exit")
    args = parser.parse_args()

    metadata_dir = Path(args.metadata_dir)
    df = load_metadata(metadata_dir)

    if args.list_unique:
        print("Unique acquisition_weather:", df["acquisition_weather"].dropna().unique())
        print("Unique acquisition_scene:", df["acquisition_scene"].dropna().unique())
        print("Unique data_source_type:", df["data_source_type"].dropna().unique())
        print("acquisition_height range:", df["acquisition_height"].min(), "to", df["acquisition_height"].max())
        return

    filtered_df = filter_metadata(
        df,
        min_height=args.min_height,
        max_height=args.max_height,
        weathers=args.weather,
        scenes=args.scene,
        data_sources=args.source
    )

    filtered_df.to_csv(args.output_csv, index=False)
    print(f"Filtered metadata saved to {args.output_csv}")
    print(f"Total rows: {len(df)} → Filtered rows: {len(filtered_df)}")

if __name__ == "__main__":
    main()