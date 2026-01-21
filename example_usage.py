import numpy as np
from pathlib import Path

from ugcd.change import detect_change
from ugcd.io import load_image, save_image
from ugcd.postprocess import postprocess_mask
from ugcd.stats import compute_change_statistics
from ugcd.vectorize import vectorize_mask
from ugcd.viz import create_overlay

#create synthetic test images if real data is not available
def create_synthetic_images():
    output_dir = Path("data")
    output_dir.mkdir(exist_ok=True)
    
    #T1: base image with some buildings
    t1 = np.zeros((500, 500), dtype=np.uint8)
    t1[100:150, 100:150] = 100  # Building 1
    t1[200:250, 200:250] = 120  # Building 2
    t1[300:350, 300:350] = 110  # Building 3
    
    #T2 same buildings + new ones (urban growth)
    t2 = t1.copy()
    t2[400:450, 400:450] = 130  # New building 1
    t2[50:100, 350:400] = 125   # New building 2
    
    noise = np.random.randint(-10, 10, t2.shape, dtype=np.int16)
    t2 = np.clip(t2.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    save_image(output_dir / "t1_synthetic.png", t1)
    save_image(output_dir / "t2_synthetic.png", t2)
    
    print(f"Created synthetic images in {output_dir}/")
    return output_dir / "t1_synthetic.png", output_dir / "t2_synthetic.png"


def main():

    t1_path = Path("data/t1.png")
    t2_path = Path("data/t2.png")
    
    if not (t1_path.exists() and t2_path.exists()):
        print("Sample images not found. Creating synthetic test images.")
        t1_path, t2_path = create_synthetic_images()
    
    print(f"Loading images: {t1_path}, {t2_path}")
    img1, meta1 = load_image(t1_path, as_grayscale=True)
    img2, meta2 = load_image(t2_path, as_grayscale=True)
    
    print(f"Image shapes: T1={img1.shape}, T2={img2.shape}")
    
    print("Detecting changes...")
    change_mask, diff_img = detect_change(img1, img2, threshold=30.0)
    
    print("Post-processing...")
    processed_mask = postprocess_mask(change_mask,min_area=100,morph_operation="closing",morph_kernel_size=3)
    
    print("Vectorizing...")
    polygons = vectorize_mask(processed_mask, simplify_tolerance=1.0)
    
    print("Computing statistics...")
    stats = compute_change_statistics(processed_mask, polygons=polygons)
    
    print("\n" + "=" * 50)
    print("Change Detection Results")
    print("=" * 50)
    print(f"Changed pixels: {stats['changed_pixels']:,} / {stats['total_pixels']:,}")
    print(f"Changed percentage: {stats['changed_percent']:.2f}%")
    print(f"Number of polygons: {stats['num_polygons']}")
    print("=" * 50)
    
    print("Creating overlay...")
    overlay = create_overlay(img2, processed_mask, alpha=0.5, color=(255, 0, 0))

    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    
    save_image(output_dir / "change_mask.png", processed_mask)
    save_image(output_dir / "overlay.png", overlay)
    
    print(f"\nOutputs saved to {output_dir}/")
    print("  - change_mask.png")
    print("  - overlay.png")


if __name__ == "__main__":
    main()





