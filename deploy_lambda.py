#!/usr/bin/env python3
"""
AWS Lambda Deployment Script for Digit Forgery Recognition
This script helps package and deploy the Lambda function to AWS.
"""

import os
import sys
import subprocess
import zipfile
import shutil
from pathlib import Path

def run_command(command, cwd=None):
    """Run a shell command and return the result"""
    try:
        result = subprocess.run(command, shell=True, cwd=cwd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Error running command: {command}")
            print(f"Error: {result.stderr}")
            return False
        return True
    except Exception as e:
        print(f"Exception running command: {command}")
        print(f"Exception: {str(e)}")
        return False

def install_dependencies():
    """Install dependencies in a temporary directory"""
    print("📦 Installing dependencies...")
    
    # Create a temporary directory for dependencies
    deps_dir = "lambda_deps"
    if os.path.exists(deps_dir):
        shutil.rmtree(deps_dir)
    os.makedirs(deps_dir)
    
    # Install dependencies
    if not run_command(f"pip install -r requirements-lambda.txt -t {deps_dir}"):
        print("❌ Failed to install dependencies")
        return False
    
    print("✅ Dependencies installed successfully")
    return True

def create_deployment_package():
    """Create the deployment package for Lambda"""
    print("📦 Creating deployment package...")
    
    # Create deployment directory
    deploy_dir = "lambda_deployment"
    if os.path.exists(deploy_dir):
        shutil.rmtree(deploy_dir)
    os.makedirs(deploy_dir)
    
    # Copy lambda function
    shutil.copy2("lambda_function.py", deploy_dir)
    
    # Copy model file
    model_file = "enhanced_mnist_forgery.keras"
    if os.path.exists(model_file):
        shutil.copy2(model_file, deploy_dir)
        print(f"✅ Model file {model_file} included")
    else:
        print(f"⚠️  Warning: Model file {model_file} not found!")
        print("   Make sure to upload the model file to S3 and modify the Lambda function to load from S3")
    
    # Copy dependencies
    deps_dir = "lambda_deps"
    if os.path.exists(deps_dir):
        for item in os.listdir(deps_dir):
            src = os.path.join(deps_dir, item)
            dst = os.path.join(deploy_dir, item)
            if os.path.isdir(src):
                shutil.copytree(src, dst)
            else:
                shutil.copy2(src, dst)
        print("✅ Dependencies copied")
    
    # Create zip file
    zip_file = "digit_forgery_lambda.zip"
    if os.path.exists(zip_file):
        os.remove(zip_file)
    
    with zipfile.ZipFile(zip_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(deploy_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arc_path = os.path.relpath(file_path, deploy_dir)
                zipf.write(file_path, arc_path)
    
    # Get file size
    file_size = os.path.getsize(zip_file) / (1024 * 1024)  # MB
    print(f"✅ Deployment package created: {zip_file} ({file_size:.1f} MB)")
    
    if file_size > 50:
        print("⚠️  Warning: Package size is large. Consider using Lambda Layers for dependencies.")
    
    return zip_file

def cleanup():
    """Clean up temporary files"""
    print("🧹 Cleaning up temporary files...")
    
    dirs_to_remove = ["lambda_deps", "lambda_deployment"]
    for dir_name in dirs_to_remove:
        if os.path.exists(dir_name):
            shutil.rmtree(dir_name)
            print(f"✅ Removed {dir_name}")

def main():
    """Main deployment function"""
    print("🚀 AWS Lambda Deployment Script for Digit Forgery Recognition")
    print("=" * 60)
    
    # Check if required files exist
    required_files = ["lambda_function.py", "requirements-lambda.txt"]
    for file in required_files:
        if not os.path.exists(file):
            print(f"❌ Required file not found: {file}")
            return False
    
    try:
        # Step 1: Install dependencies
        if not install_dependencies():
            return False
        
        # Step 2: Create deployment package
        zip_file = create_deployment_package()
        if not zip_file:
            return False
        
        print("\n" + "=" * 60)
        print("✅ Deployment package ready!")
        print(f"📦 Package: {zip_file}")
        print("\n📋 Next steps:")
        print("1. Upload the zip file to AWS Lambda")
        print("2. Set the handler to 'lambda_function.lambda_handler'")
        print("3. Configure memory (recommended: 1024 MB)")
        print("4. Set timeout (recommended: 30 seconds)")
        print("5. Create an API Gateway if you want HTTP access")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during deployment: {str(e)}")
        return False
    
    finally:
        cleanup()

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
