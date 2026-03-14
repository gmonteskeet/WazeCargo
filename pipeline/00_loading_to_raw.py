"""
WazeCargo S3 Raw Layer Upload Script
====================================
Uploads CSV files (ingresos/salidas) to S3 raw layer.

Usage:
    python upload_raw.py <base_folder> --type <ingresos|salidas> --year <year|all> [--force]

Examples:
    # Windows
    python upload_raw.py "C:\\Users\\user\\data" --type ingresos --year 2024
    
    # Mac/Linux
    python upload_raw.py "/home/user/data" --type ingresos --year 2024
    
    # Current folder
    python upload_raw.py "." --type ingresos --year all
    
    # Force overwrite (for incomplete 2026 file)
    python upload_raw.py "." --type ingresos --year 2026 --force
"""

import sys
import os
import argparse
import boto3
from botocore.exceptions import ClientError

# =============================================================================
# CONFIGURATION
# =============================================================================

BUCKET_NAME = "wazecargo-263704545424-eu-north-1-an"
REGION = "eu-north-1"
VALID_YEARS = list(range(2000, 2027))  # 2020 to 2026
VALID_TYPES = ["ingresos", "salidas"]


# =============================================================================
# PATH UTILITIES
# =============================================================================

def get_path_separator(path):
    """Detect path separator based on path string."""
    if "\\" in path:
        return "\\"  # Windows
    return "/"  # Mac/Linux


def join_path(base, *parts):
    """Join path parts using the appropriate separator."""
    sep = get_path_separator(base)
    return sep.join([base] + list(parts))


def build_local_path(base_folder, file_type, year):
    """Build local file path for a given type and year."""
    if file_type == "salidas":
        filename = f"{file_type}_{year}.csv"  # salidas_2020.csv
    else:
        filename = f"{file_type}{year}.csv"   # ingresos2020.csv
    return join_path(base_folder, filename)


def build_s3_key(file_type, year):
    """Build S3 key for a given type and year."""
    if file_type == "salidas":
        filename = f"{file_type}_{year}.csv"  # salidas_2020.csv
    else:
        filename = f"{file_type}{year}.csv"   # ingresos2020.csv
    return f"raw/{file_type}/year={year}/{filename}"


# =============================================================================
# S3 OPERATIONS
# =============================================================================

def get_s3_client():
    """Create S3 client."""
    return boto3.client("s3", region_name=REGION)


def file_exists_in_s3(s3_client, key):
    """Check if file already exists in S3."""
    try:
        s3_client.head_object(Bucket=BUCKET_NAME, Key=key)
        return True
    except ClientError as e:
        if e.response["Error"]["Code"] == "404":
            return False
        raise


def upload_file_to_s3(s3_client, local_path, s3_key):
    """Upload a single file to S3."""
    try:
        s3_client.upload_file(local_path, BUCKET_NAME, s3_key)
        return True
    except ClientError as e:
        print(f"  ERROR: Failed to upload: {e}")
        return False


# =============================================================================
# MAIN UPLOAD LOGIC
# =============================================================================

def upload_single_file(s3_client, base_folder, file_type, year, force=False):
    """Upload a single CSV file to S3 raw layer."""
    local_path = build_local_path(base_folder, file_type, year)
    s3_key = build_s3_key(file_type, year)
    
    print(f"\nProcessing: {file_type}{year}.csv")
    print(f"  Local: {local_path}")
    print(f"  S3:    s3://{BUCKET_NAME}/{s3_key}")
    
    # Check if local file exists
    if not os.path.exists(local_path):
        print(f"  SKIP: Local file not found")
        return False
    
    # Check if already exists in S3
    if file_exists_in_s3(s3_client, s3_key):
        if force:
            print(f"  OVERWRITE: File exists in S3, --force flag used")
        else:
            print(f"  SKIP: File already exists in S3 (use --force to overwrite)")
            return False
    
    # Upload
    print(f"  UPLOADING...")
    if upload_file_to_s3(s3_client, local_path, s3_key):
        file_size = os.path.getsize(local_path) / 1024
        print(f"  SUCCESS: Uploaded ({file_size:.1f} KB)")
        return True
    
    return False


def upload_files(base_folder, file_type, years, force=False):
    """Upload multiple files to S3."""
    print("=" * 60)
    print("WazeCargo S3 Raw Layer Upload")
    print("=" * 60)
    print(f"Bucket:  {BUCKET_NAME}")
    print(f"Region:  {REGION}")
    print(f"Type:    {file_type}")
    print(f"Years:   {years}")
    print(f"Force:   {force}")
    print(f"Source:  {base_folder}")
    
    s3_client = get_s3_client()
    
    uploaded = 0
    skipped = 0
    failed = 0
    
    for year in years:
        result = upload_single_file(s3_client, base_folder, file_type, year, force)
        if result:
            uploaded += 1
        elif os.path.exists(build_local_path(base_folder, file_type, year)):
            skipped += 1
        else:
            failed += 1
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Uploaded: {uploaded}")
    print(f"Skipped:  {skipped}")
    print(f"Not found: {failed}")
    
    return uploaded > 0 or skipped > 0


# =============================================================================
# ARGUMENT PARSING
# =============================================================================

def print_usage():
    """Print usage instructions."""
    print()
    print("=" * 60)
    print("WazeCargo S3 Raw Layer Upload Script")
    print("=" * 60)
    print()
    print("Usage:")
    print("  python upload_raw.py <base_folder> --type <type> --year <year> [--force]")
    print()
    print("Arguments:")
    print("  base_folder  Path to folder containing CSV files")
    print("  --type       File type: ingresos or salidas")
    print("  --year       Year (2020-2026) or 'all'")
    print("  --force      Overwrite existing files in S3")
    print()
    print("Examples:")
    print()
    print("  Windows:")
    print('    python upload_raw.py "C:\\Users\\user\\data" --type ingresos --year 2024')
    print()
    print("  Mac/Linux:")
    print('    python upload_raw.py "/home/user/data" --type ingresos --year 2024')
    print()
    print("  Current folder, all years:")
    print('    python upload_raw.py "." --type ingresos --year all')
    print()
    print("  Force overwrite (for 2026):")
    print('    python upload_raw.py "." --type ingresos --year 2026 --force')
    print()


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Upload CSV files to S3 raw layer",
        add_help=False
    )
    parser.add_argument("base_folder", nargs="?", help="Path to folder containing CSV files")
    parser.add_argument("--type", dest="file_type", choices=VALID_TYPES, help="File type")
    parser.add_argument("--year", dest="year", help="Year (2020-2026) or 'all'")
    parser.add_argument("--force", action="store_true", help="Overwrite existing files")
    parser.add_argument("-h", "--help", action="store_true", help="Show help")
    
    args = parser.parse_args()
    
    # Show help if requested or missing arguments
    if args.help or not args.base_folder or not args.file_type or not args.year:
        print_usage()
        sys.exit(0 if args.help else 1)
    
    # Validate base folder
    if not os.path.isdir(args.base_folder):
        print(f"ERROR: Folder not found: {args.base_folder}")
        sys.exit(1)
    
    # Parse years
    if args.year.lower() == "all":
        years = VALID_YEARS
    else:
        try:
            year = int(args.year)
            if year not in VALID_YEARS:
                print(f"ERROR: Year must be between 2020 and 2026")
                sys.exit(1)
            years = [year]
        except ValueError:
            print(f"ERROR: Invalid year: {args.year}")
            sys.exit(1)
    
    return args.base_folder, args.file_type, years, args.force


# =============================================================================
# MAIN
# =============================================================================

def main():
    base_folder, file_type, years, force = parse_arguments()
    success = upload_files(base_folder, file_type, years, force)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()