from ultralytics import YOLO
import cv2
import os


def yolo_label_image(
    image_path: str,
    model_path: str,
    output_dir: str = "labeled_images",
    conf_thres: float = 0.25,
):
    """Run YOLO on a single image and save an annotated copy.

    All paths can be relative to this file or absolute.
    """

    # Resolve paths relative to this script so it works when run from anywhere
    script_dir = os.path.dirname(os.path.abspath(__file__))

    if not os.path.isabs(image_path):
        image_path = os.path.join(script_dir, image_path)
    if not os.path.isabs(model_path):
        model_path = os.path.join(script_dir, model_path)
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(script_dir, output_dir)

    os.makedirs(output_dir, exist_ok=True)

    # Load YOLO model
    print(f"Loading model from: {model_path}")
    model = YOLO(model_path)

    # Read image
    print(f"Reading image from: {image_path}")
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found or cannot be read: {image_path}")

    # Run inference
    print("Running inference...")
    results = model.predict(source=img, conf=conf_thres, verbose=False)

    if not results:
        raise RuntimeError("No results returned by the model.")

    print(f"Results: {len(results)} objects found.")
    result = results[0]
    print(f"Detections: {len(result.boxes)} objects found.")
    # Plot detections on the image
    annotated_img = result.plot()

    # Build output path
    base_name = os.path.basename(image_path)
    out_path = os.path.join(output_dir, f"labeled_{base_name}")

    # Save annotated image
    cv2.imwrite(out_path, annotated_img)
    print(f"Annotated image saved to: {out_path}")

    return out_path


# -------------------- RUN --------------------
if __name__ == "__main__":
    output_image_path = yolo_label_image(
        image_path="test4.jpg",          # input image (relative to this file)
        model_path="best.pt",            # your trained model (relative to this file)
        output_dir="outputs",            # save folder (relative to this file)
        conf_thres=0.25,
    )
    print(f"Done. Labeled image: {output_image_path}")
