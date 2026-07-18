# Vehicle Detection Suite - Web UI

This directory contains a completely client-side web application for vehicle and object detection using YOLOv8, running directly in the browser via **ONNX Runtime Web**.

Because the machine learning inference is executed locally by the browser (leveraging WebAssembly / GPU acceleration), there are **zero hosting costs**, **no server quotas**, and **100% privacy** (your images/videos never leave your computer).

---

## Getting Started

### 1. Export your YOLOv8 Model to ONNX
Run the python exporter script in the project root to generate the browser-compatible `.onnx` model weights:
```bash
python export_onnx.py
```
This script locates your trained weights (e.g., `runs/detect/train6/weights/best.pt`) and converts them to `runs/detect/train6/weights/best.onnx`.

### 2. Copy the Model to the Web UI Directory
To bundle your custom model with the static website, copy the exported `.onnx` file into this `web_ui` folder and rename it to `best.onnx`:
* **Source:** `runs/detect/train6/weights/best.onnx`
* **Destination:** `web_ui/best.onnx`

*Note: If you don't copy the file, you can still upload the `.onnx` file directly from your hard drive using the "Upload Custom .onnx Model" button in the web browser!*

### 3. Run Locally
Web browsers block Loading WebAssembly models directly from `file://` URIs due to security policies. You must run a lightweight local web server to test the interface.

Here are a few quick ways to launch a local server:

#### Option A: Python (Built-in)
Run this command in the `web_ui` folder:
```bash
python -m http.server 8000
```
Then open [http://localhost:8000](http://localhost:8000) in your browser.

#### Option B: VS Code Live Server
If you use VS Code, right-click `index.html` and select **"Open with Live Server"**.

---

## Free Public Deployments (No Quota, Free Forever)

Since this is a static site (HTML, CSS, JS), you can host it for free on premium static providers. Here are the 3 best methods to get a public URL:

### Method 1: Netlify Drop (Easiest, No Code/Git Required)
1. Open your browser and go to [Netlify Drop](https://app.netlify.com/drop).
2. Drag and drop the entire `web_ui` folder onto the web page.
3. Your site will deploy in **5 seconds** and give you a free, public URL!
4. *(Optional)* Register a free account to change the URL slug or attach a custom domain.

### Method 2: GitHub Pages
1. Push this project code to a GitHub repository.
2. Go to the repository **Settings** tab.
3. Click on **Pages** in the left sidebar.
4. Under **Build and deployment**, set the source to **Deploy from a branch**.
5. Select the `main` or `master` branch and the folder where your static files live, then click **Save**.
6. GitHub will deploy the site to a public URL: `https://<username>.github.io/<repo-name>/`.

### Method 3: Vercel CLI
Run this command inside the `web_ui` directory:
```bash
npx vercel
```
Follow the prompts (log in with a free account), and Vercel will instantly deploy your static folder and give you a production-ready, secure public URL for free.
