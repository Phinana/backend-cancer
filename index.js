const express = require("express");
const multer = require("multer");
const tf = require("@tensorflow/tfjs-node");
const { v4: uuidv4 } = require("uuid");
const path = require("path");

const app = express();
const PORT = 3000;

// URL langsung untuk model
// app.use('/model', express.static("file://./submission-model/model.json"));
// app.listen(3001, () => {
//   console.log("File server running on http://localhost:3000");
// });

const MODEL_PATH = path.resolve('/mnt/d/Tugas/Dicoding-Penerapan-Machine-Learning', 'submissions-model', 'model.json');


// Configure Multer for file uploads
const upload = multer({
  limits: { fileSize: 1000000 }, // 1MB limit
  fileFilter(req, file, cb) {
    if (!file.mimetype.startsWith("image/")) {
      return cb(new Error("Only image files are allowed!"));
    }
    cb(null, true);
  },
});

// Load TensorFlow model
let model;
const loadModel = async () => {
  try {
    model = await tf.loadGraphModel(`file://${MODEL_PATH}`);
    console.log("Model loaded successfully.");
  } catch (error) {
    console.error("Error loading model:", error);
  }
};
loadModel();

// Endpoint: Predict
app.post("/predict", upload.single("image"), async (req, res) => {
  console.log(req.file);
  try {
    if (!model) throw new Error("Model not loaded.");
    const imageBuffer = req.file.buffer;
    const tensor = tf.node
      .decodeImage(imageBuffer)
      .resizeBilinear([224, 224])
      .expandDims(0);
    const prediction = model.predict(tensor);
    const probability = prediction.arraySync()[0][0];

    const id = uuidv4();
    const result = probability > 0.5 ? "Cancer" : "Non-cancer";
    const suggestion =
      result === "Cancer"
        ? "Segera periksa ke dokter!"
        : "Penyakit kanker tidak terdeteksi.";
    const response = {
      status: "success",
      message: "Model is predicted successfully",
      data: {
        id,
        result,
        suggestion,
        createdAt: new Date().toISOString(),
      },
    };

    res.json(response);
  } catch (error) {
    if (
      error.message ===
      "Payload content length greater than maximum allowed: 1000000"
    ) {
      res.status(413).json({
        status: "fail",
        message: "Payload content length greater than maximum allowed: 1000000",
      });
    } else {
      res.status(400).json({
        status: "fail",
        message: "Terjadi kesalahan dalam melakukan prediksi",
      });
    }
  }
});

// Error handler for file uploads
app.use((error, req, res, next) => {
  if (
    error instanceof multer.MulterError ||
    error.message === "Only image files are allowed!"
  ) {
    return res.status(400).json({ status: "fail", message: error.message });
  }
  next(error);
});

// Start server
app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});
