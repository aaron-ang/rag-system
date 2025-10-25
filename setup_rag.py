"""
Setup script for the Medical RAG System
This script will start Qdrant, process data, and set up the vector database.
"""

import subprocess
import time
import os
import sys
from dotenv import load_dotenv

load_dotenv()


def run_command(command, description):
    """Run a command and handle errors."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error in {description}: {e}")
        print(f"Error output: {e.stderr}")
        return False


def check_docker():
    """Check if Docker is installed and running."""
    print("🐳 Checking Docker...")
    try:
        subprocess.run("docker --version", shell=True, check=True, capture_output=True)
        subprocess.run("docker-compose --version", shell=True, check=True, capture_output=True)
        print("✅ Docker is available")
        return True
    except subprocess.CalledProcessError:
        print("❌ Docker not found. Please install Docker Desktop.")
        return False


def start_qdrant():
    """Start Qdrant using Docker Compose."""
    print("🚀 Starting Qdrant...")
    
    # Stop any existing containers
    subprocess.run("docker-compose down", shell=True, capture_output=True)
    
    # Start Qdrant
    if run_command("docker-compose up -d", "Starting Qdrant container"):
        print("⏳ Waiting for Qdrant to be ready...")
        time.sleep(10)  # Wait for Qdrant to start
        
        # Test connection
        try:
            from qdrant_client import QdrantClient
            client = QdrantClient(host="localhost", port=6333)
            client.get_collections()
            print("✅ Qdrant is running and accessible")
            return True
        except Exception as e:
            print(f"❌ Qdrant connection failed: {e}")
            return False
    return False


def setup_environment():
    """Set up environment variables."""
    print("🔧 Setting up environment...")
    
    env_file = ".env"
    if not os.path.exists(env_file):
        print("📝 Creating .env file...")
        with open(env_file, "w") as f:
            f.write("# OpenAI API Key (optional, for AI-generated answers)\n")
            f.write("OPENAI_API_KEY=your_openai_api_key_here\n")
            f.write("\n# Qdrant Configuration\n")
            f.write("QDRANT_HOST=localhost\n")
            f.write("QDRANT_PORT=6333\n")
        print("✅ Created .env file. Please add your OpenAI API key if you want AI-generated answers.")
    else:
        print("✅ .env file already exists")


def main():
    """Main setup function."""
    print("🏥 Medical RAG System Setup")
    print("=" * 50)
    
    # Check Docker
    if not check_docker():
        print("\n❌ Please install Docker Desktop and try again.")
        print("Download from: https://www.docker.com/products/docker-desktop")
        return
    
    # Setup environment
    setup_environment()
    
    # Start Qdrant
    if not start_qdrant():
        print("\n❌ Failed to start Qdrant. Please check Docker and try again.")
        return
    
    print("\n🎉 Setup complete!")
    print("\nNext steps:")
    print("1. Run: python rag_system.py  # Process data and create vector database")
    print("2. Run: python query_interface.py  # Start interactive query interface")
    print("\nOptional:")
    print("- Add your OpenAI API key to .env file for AI-generated answers")
    print("- Run: docker-compose down  # Stop Qdrant when done")


if __name__ == "__main__":
    main()
