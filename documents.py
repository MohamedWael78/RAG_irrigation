"""
01_documents.py – Step 1: Load PDF documents, extract text AND images.

Extracts text page-by-page (with --- PAGE N --- markers) and extracts images
from each PDF page. Images are saved to data/images/ with metadata tracked in
data/images/image_metadata.json.

Images are filtered by minimum size (>= 100x100 px) to skip tiny logos/icons.

Run as:  python 01_documents.py
"""

import os
import json

import fitz  # PyMuPDF

PDF_DIR = os.path.join("data", "pdfs")
RAW_DIR = os.path.join("data", "raw")
IMAGE_DIR = os.path.join("data", "images")
os.makedirs(PDF_DIR, exist_ok=True)
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(IMAGE_DIR, exist_ok=True)

# Minimum image dimensions to extract (skip tiny icons/logos/decorations)
MIN_IMAGE_WIDTH = 100
MIN_IMAGE_HEIGHT = 100

METADATA_FILE = os.path.join(IMAGE_DIR, "image_metadata.json")


# ═══════════════════════════════════════════════════════════════════════════
# TEXT EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from a PDF page by page using PyMuPDF."""
    doc = fitz.open(pdf_path)
    pages_text = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text")
        if text.strip():
            pages_text.append(f"--- PAGE {page_num + 1} ---\n{text.strip()}")
    doc.close()
    return "\n\n".join(pages_text)


# ═══════════════════════════════════════════════════════════════════════════
# IMAGE EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════

def extract_images_from_pdf(pdf_path: str) -> list[dict]:
    """Extract meaningful images from a PDF file.

    For each page, extracts all embedded images, filters by minimum size,
    saves them to data/images/, and returns metadata dict for each image.

    Returns list of dicts:
        {"path", "filename", "source_pdf", "page", "image_index",
         "width", "height", "ext"}
    """
    doc = fitz.open(pdf_path)
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
    all_images = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        image_list = page.get_images(full=True)

        for img_idx, img_info in enumerate(image_list):
            xref = img_info[0]

            try:
                base_image = doc.extract_image(xref)
            except Exception:
                continue

            if not base_image or not base_image.get("image"):
                continue

            width = base_image.get("width", 0)
            height = base_image.get("height", 0)

            # Skip tiny images (icons, logos, decorative elements)
            if width < MIN_IMAGE_WIDTH or height < MIN_IMAGE_HEIGHT:
                continue

            image_bytes = base_image["image"]
            image_ext = base_image["ext"]  # png, jpeg, etc.

            image_filename = f"{pdf_name}_page{page_num + 1}_img{img_idx + 1}.{image_ext}"
            image_path = os.path.join(IMAGE_DIR, image_filename)

            with open(image_path, "wb") as f:
                f.write(image_bytes)

            all_images.append({
                "path": image_path,
                "filename": image_filename,
                "source_pdf": os.path.basename(pdf_path),
                "page": page_num + 1,
                "image_index": img_idx + 1,
                "width": width,
                "height": height,
                "ext": image_ext,
                "size_bytes": len(image_bytes),
            })

    doc.close()
    return all_images


def save_image_metadata(images: list[dict]) -> str:
    """Save image metadata list to JSON file."""
    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(images, f, indent=2, ensure_ascii=False)
    return METADATA_FILE


def load_image_metadata() -> list[dict]:
    """Load previously saved image metadata."""
    if not os.path.isfile(METADATA_FILE):
        return []
    with open(METADATA_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


# ═══════════════════════════════════════════════════════════════════════════
# ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════════

def load_pdf_documents() -> list[str]:
    """Scan data/pdfs/ for PDF files, extract text + images, save to data/raw/
    and data/images/.

    Returns list of created .txt file paths.
    """
    pdf_files = sorted(
        f for f in os.listdir(PDF_DIR) if f.lower().endswith(".pdf")
    )

    if not pdf_files:
        print("⚠️  No PDF files found in data/pdfs/.")
        print("   Place your irrigation PDF documents there and re-run.")
        print("   Falling back to built-in sample documents ...")
        return create_sample_documents()

    all_image_metadata = []
    filepaths = []

    for pdf_file in pdf_files:
        pdf_path = os.path.join(PDF_DIR, pdf_file)

        # ── Extract text ──
        try:
            text = extract_text_from_pdf(pdf_path)
            if text.strip():
                raw_filename = os.path.splitext(pdf_file)[0] + ".txt"
                raw_path = os.path.join(RAW_DIR, raw_filename)
                with open(raw_path, "w", encoding="utf-8") as f:
                    f.write(text)
                filepaths.append(raw_path)
                num_pages = text.count("--- PAGE")
                print(f"✅ Text: {pdf_file} → {raw_filename}  ({num_pages} pages, {len(text)} chars)")
            else:
                print(f"⚠️  {pdf_file}: no extractable text (scanned/image PDF)")
        except Exception as e:
            print(f"❌ Text extraction failed for {pdf_file}: {e}")

        # ── Extract images ──
        try:
            images = extract_images_from_pdf(pdf_path)
            all_image_metadata.extend(images)
            if images:
                print(f"✅ Images: {pdf_file} → {len(images)} images extracted")
            else:
                print(f"   {pdf_file}: no meaningful images found (or all below {MIN_IMAGE_WIDTH}x{MIN_IMAGE_HEIGHT}px)")
        except Exception as e:
            print(f"❌ Image extraction failed for {pdf_file}: {e}")

    # Save combined image metadata
    if all_image_metadata:
        save_image_metadata(all_image_metadata)
        print(f"\n📷 Total images extracted: {len(all_image_metadata)}")
        print(f"   Metadata saved to: {METADATA_FILE}")

    if not filepaths:
        print("⚠️  All PDFs failed. Using sample documents as fallback.")
        return create_sample_documents()

    return filepaths


# ═══════════════════════════════════════════════════════════════════════════
# BUILT-IN SAMPLE DOCUMENTS (fallback)
# ═══════════════════════════════════════════════════════════════════════════

SAMPLE_DOCUMENTS = {
    "fao_drip_irrigation_design.txt": """
FAO Irrigation Manual - Drip Irrigation System Design (Excerpt)

--- PAGE 1 ---
Chapter 1: Introduction to Drip Irrigation
Drip irrigation delivers water directly to the root zone through valves, pipes, tubing, and emitters.
Achieving water application efficiencies of 90-95% compared to 50-70% for conventional methods.
FAO Paper 56: properly designed drip systems reduce water use by 30-50% while maintaining yields.

--- PAGE 2 ---
Chapter 2: System Components
- Control head: pump, pressure regulator, main filter (120-200 mesh), fertigation unit
- Main line: PVC/PE pipe, 50-110 mm diameter
- Sub-main / manifold: 25-50 mm pipes
- Lateral lines: 12-20 mm polyethylene tubes
- Emitters / drippers: discharge 1-8 L/h. Types: point-source (trees), inline (row crops), micro-sprayers

--- PAGE 3 ---
Chapter 3: Emitter Spacing by Soil Type
  Soil Type          | Emitter Spacing (m) | Emitter Flow (L/h) | Wetted Diameter (m)
  Sandy              | 0.3 - 0.5           | 2 - 4              | 0.3 - 0.5
  Loamy Sand         | 0.5 - 0.7           | 2 - 4              | 0.5 - 0.8
  Sandy Loam         | 0.6 - 0.9           | 2 - 4              | 0.6 - 1.0
  Loam               | 0.8 - 1.2           | 1 - 2              | 0.8 - 1.2
  Silt Loam          | 1.0 - 1.5           | 1 - 2              | 1.0 - 1.5
  Clay Loam          | 1.2 - 2.0           | 0.5 - 1            | 1.5 - 2.5
  Clay               | 2.0 - 3.0           | 0.5 - 1            | 2.0 - 3.5

--- PAGE 4 ---
Chapter 4: Hydraulic Design
  Q_total = N_emitters x q_emitter
  Example: 4000 trees, 2 drippers per tree at 4 L/h: Q_total = 4000 x 2 x 4 = 32000 L/h = 8.9 L/s
  V_day = Q_total x t. Operating 4h: V_day = 32000 x 4 = 128000 L = 128 m3
  Pressure: 1.0-2.5 bar at emitter. Variation < +/-10% for DU > 90%.

--- PAGE 5 ---
Chapter 5: Filtration and Maintenance
- Primary: 120-200 mesh screen/disc filter
- Secondary: sand media filter for organic-rich water (algae > 15 mg/L)
- Weekly: check filters, flush laterals, inspect gauges
- Monthly: catch-can uniformity test, clean injectors
- Seasonally: flush system, replace filters, check leaks
- Annually: full audit, replace degraded emitters
""",
    "crop_water_requirements.txt": """
FAO Crop Water Requirements (Excerpt)

--- PAGE 1 ---
Chapter 1: Crop Coefficient (Kc) Approach
  ETc = Kc x ET0
FAO-56 Kc: initial (Kc_ini), mid-season (Kc_mid), late-season (Kc_end)

--- PAGE 2 ---
Chapter 2: Kc Values for Major Crops
  Crop                | Kc_initial | Kc_mid  | Kc_late | Root Depth (m) | Height (m)
  Tomato              | 0.60       | 1.15    | 0.80    | 0.6-1.0        | 0.6-1.2
  Wheat               | 0.30       | 1.15    | 0.25    | 1.0-1.5        | 1.0
  Maize / Corn        | 0.30       | 1.20    | 0.50    | 1.0-1.7        | 2.0-2.5
  Rice                | 1.05       | 1.20    | 0.90    | 0.5-1.0        | 1.0
  Cotton              | 0.35       | 1.20    | 0.65    | 1.0-1.5        | 1.2-1.5
  Potato              | 0.40       | 1.10    | 0.75    | 0.4-0.6        | 0.3-0.5
  Onion               | 0.50       | 1.05    | 0.75    | 0.3-0.6        | 0.3-0.5
  Citrus              | 0.70       | 0.85    | 0.65    | 1.0-1.5        | 3.0-5.0
  Grapevine           | 0.30       | 0.70    | 0.45    | 1.0-2.0        | 1.5-2.0
  Olive               | 0.55       | 0.65    | 0.55    | 1.2-1.7        | 3.0-5.0
  Alfalfa             | 0.40       | 1.20    | 1.05    | 1.0-2.0        | 0.7
  Lettuce             | 0.30       | 1.00    | 0.90    | 0.3-0.5        | 0.3
  Sunflower           | 0.35       | 1.15    | 0.35    | 1.0-1.5        | 2.0

--- PAGE 3 ---
Chapter 3: Growth Stage Duration (days)
  Tomato: initial 25-35, dev 40-50, mid 50-70, late 30-40, total 145-195
  Maize: initial 20-30, dev 30-40, mid 40-50, late 20-30, total 110-150

--- PAGE 4 ---
Chapter 4: Effective Rainfall
  Ptotal < 70 mm/month -> Reff = 0.6 x Ptotal
  Ptotal 70-200 -> Reff = 0.8 x Ptotal - 20
  IRn = ETc - Reff
  Example: Tomato mid, Kc=1.15, ET0=5.2, P=40mm -> IRn = 155.4 mm/month

--- PAGE 5 ---
Chapter 5: Soil Water Balance
  TAW = 1000 x (theta_FC - theta_WP) x Zr
  RAW = p x TAW (p=0.3-0.6)
  Loam: TAW=120mm, RAW=60mm for tomato (p=0.5)
""",
    "soil_properties_guide.txt": """
Soil Water Properties (Excerpt)

--- PAGE 1 ---
Soil Texture and Classification
  Sandy: 85-100% sand, high infiltration, low retention
  Loamy Sand: 70-85% sand, rapid drainage
  Sandy Loam: 50-70% sand, moderate infiltration
  Loam: 30-50% sand, balanced
  Silt Loam: 10-30% sand, good retention
  Clay Loam: 20-35% sand, slow infiltration
  Clay: 0-30% sand, very slow infiltration

--- PAGE 2 ---
Field Capacity, Wilting Point, AWC
  Sandy: FC=0.07-0.11, WP=0.02-0.04, AWC=50-70 mm/m
  Sandy Loam: FC=0.18-0.24, WP=0.06-0.10, AWC=120-140 mm/m
  Loam: FC=0.25-0.32, WP=0.10-0.15, AWC=150-170 mm/m
  Clay: FC=0.39-0.48, WP=0.20-0.28, AWC=190-200 mm/m

--- PAGE 3 ---
Infiltration Rates (mm/h)
  Sandy: 50-200, Loamy Sand: 30-80, Sandy Loam: 20-60
  Loam: 10-30, Silt Loam: 5-20, Clay Loam: 1-10, Clay: 0.5-5

--- PAGE 4 ---
Root Zone Depths (m)
  Lettuce: 0.3-0.5, Tomato: 0.6-1.0, Wheat: 0.8-1.2
  Maize: 1.0-1.5, Cotton: 1.0-1.5, Citrus: 1.0-1.5

--- PAGE 5 ---
Soil Moisture 100% But Plants Wilt - Causes
1. Sensor too close to emitter -> move 15-25 cm away
2. Root rot / waterlogging -> reduce frequency, improve drainage
3. Salinity buildup -> check EC > 4 dS/m, leach periodically
4. Sensor calibration error -> calibrate with gravimetric samples
5. Shallow root zone -> sensor below active root depth
""",
    "irrigation_troubleshooting.txt": """
Irrigation Troubleshooting (Excerpt)

--- PAGE 1 ---
Emitter Clogging
- Physical: sand, silt, algae, insect debris
- Chemical: calcium carbonate, iron oxide precipitation
- Biological: bacterial slime, root intrusion
Detection: flow rate test (uniformity < 85% = clogging)
Prevention: 120-200 mesh filtration, weekly flush, acid injection (pH 2-3 monthly), chlorine 1-2 ppm

--- PAGE 2 ---
Pressure Problems
- Undersized lateral -> resize (16->20mm)
- Failed pressure regulator -> install 1.5-2.0 bar
- Elevation changes -> PC emitters or pressure-reducing valves
- Air locks -> air/vacuum release valves at high points

--- PAGE 3 ---
Valve Troubleshooting (Hunter PGV, Rain Bird)
Won't open: check solenoid (20-60 ohm), 24 VAC, clean diaphragm, clear bleed port
Won't close: clean diaphragm seat, replace stuck solenoid, ensure min 1.0 bar, close manual bleed
Partial opening: check diaphragm tears, install water hammer arrestors

--- PAGE 4 ---
Catch-Can Test: DU = Avg(lowest 25%) / Avg(all)
DU > 90%: Excellent, 80-90%: Acceptable, 70-80%: Poor, < 70%: Unacceptable
Causes of poor DU: mixed emitter types, pressure variation > +/-10%, partial clogging, wrong spacing

--- PAGE 5 ---
Maintenance Checklist
Pre-season: clean filters, flush system 3-5 min, check pressures, test valves
Monthly: catch-can test, clean filters, inspect 20 random emitters
Post-season: full flush, replace damaged parts, drain pump
""",
    "eto_calculation_methods.txt": """
ET0 Calculation Methods (Excerpt)

--- PAGE 1 ---
Penman-Monteith FAO-56 (standard method)
  ET0 = [0.408 x Delta x (Rn-G) + gamma x (900/(T+273)) x u2 x (es-ea)] / [Delta + gamma x (1+0.34 x u2)]
  Requires: Tmax, Tmin, RHmax, RHmin, solar radiation, wind speed, altitude

--- PAGE 2 ---
Hargreaves-Samani (simplified, temperature-only)
  ET0 = 0.0023 x (Tmean+17.8) x (Tmax-Tmin)^0.5 x Ra
  Accuracy: +/-15-20% of Penman-Monteith

--- PAGE 3 ---
ET0 by Climate Zone
  Humid tropical: 2-4 mm/day, 800-1500 mm/year
  Subtropical: 3-5 mm/day, 1200-1800 mm/year
  Mediterranean: 4-7 mm/day, 1500-2300 mm/year
  Arid desert: 7-11 mm/day, 2500-3500 mm/year

--- PAGE 4 ---
ET0 by Latitude
  0-15: summer 4.0-5.5, winter 3.0-4.5 mm/day
  15-30: summer 5.5-8.0, winter 2.0-4.0
  30-45: summer 5.0-7.5, winter 1.0-3.0
  Cairo (30.04N): Jan 1.8, Apr 4.6, Jul 7.2, Oct 3.8 mm/day

--- PAGE 5 ---
Irrigation Scheduling with ET0
  ETc = Kc x ET0
  Tomato mid-season Cairo July: ETc = 1.15 x 7.2 = 8.28 mm/day
  Weekly volume = 8.28 x 7 x 10000 = 579600 L/ha
  Q = 579600 / (4 x 7) = 20700 L/h/ha
""",
}


def create_sample_documents() -> list[str]:
    """Write built-in sample documents (fallback when no PDFs)."""
    filepaths = []
    for filename, content in SAMPLE_DOCUMENTS.items():
        filepath = os.path.join(RAW_DIR, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content.strip())
        filepaths.append(filepath)
        print(f"✅ Sample: {filepath}  ({len(content.strip())} chars)")
    # No images for sample documents
    save_image_metadata([])
    return filepaths


def create_documents() -> list[str]:
    """Main entry: load PDFs (text + images) or create samples."""
    raw_existing = [
        f for f in os.listdir(RAW_DIR) if f.endswith(".txt")
    ] if os.path.isdir(RAW_DIR) else []

    if raw_existing:
        print(f"📄 {len(raw_existing)} raw documents already exist.")
        print("   To force reload, delete data/raw/ and data/pdfs/ first.")
        return sorted(os.path.join(RAW_DIR, f) for f in raw_existing)

    return load_pdf_documents()


def list_raw_documents() -> list[str]:
    if not os.path.isdir(RAW_DIR):
        return []
    return sorted(os.path.join(RAW_DIR, f) for f in os.listdir(RAW_DIR) if f.endswith(".txt"))


def list_extracted_images() -> list[dict]:
    """Return list of all extracted image metadata dicts."""
    return load_image_metadata()


if __name__ == "__main__":
    paths = create_documents()
    images = load_image_metadata()
    print(f"\n📝 {len(paths)} raw documents in {RAW_DIR}/")
    print(f"📷 {len(images)} images in {IMAGE_DIR}/")
    for p in paths:
        print(f"   → {p}")
    for img in images[:5]:
        print(f"   → 📷 {img['filename']} (p.{img['page']}, {img['width']}x{img['height']})")