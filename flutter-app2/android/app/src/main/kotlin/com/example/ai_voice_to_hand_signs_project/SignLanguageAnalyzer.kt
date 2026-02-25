package com.example.ai_voice_to_hand_signs_project

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Matrix
import android.os.SystemClock
import android.util.Log
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.framework.image.MPImage
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarker
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarkerResult
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import java.util.LinkedList
import kotlin.math.max
import kotlin.math.sqrt

class SignLanguageAnalyzer(
    private val context: Context,
    private val onResult: (String, Float) -> Unit
) : ImageAnalysis.Analyzer {

    private var handLandmarker: HandLandmarker? = null
    private var tflite: Interpreter? = null
    
    // Buffer for LSTM: 60 frames, 63 features (21 landmarks * 3 coords)
    private val frameBuffer = LinkedList<FloatArray>()
    private val WINDOW_SIZE = 60
    private val NUM_FEATURES = 63 // 21 * 3
    
    // Labels from label_map.json - 29 classes matching trained model
    private val labels = listOf(
        "Teacher", "I", "have", "question", "a"
    )

    init {
        setupMediaPipe()
        setupTFLite()
    }

    private fun setupMediaPipe() {
        val baseOptions = BaseOptions.builder()
            .setModelAssetPath("hand_landmarker.task")
            .build()

        val options = HandLandmarker.HandLandmarkerOptions.builder()
            .setBaseOptions(baseOptions)
            .setNumHands(1)
            .setMinHandDetectionConfidence(0.5f)
            .setMinHandPresenceConfidence(0.5f)
            .setMinTrackingConfidence(0.5f)
            .setRunningMode(RunningMode.IMAGE) // Using IMAGE mode for simplicity in Analyzer
            .build()

        handLandmarker = HandLandmarker.createFromOptions(context, options)
    }

    private fun setupTFLite() {
        try {
            val modelBuffer = loadModelFile("sign_language_model.tflite")
            val options = Interpreter.Options()
            
            // Use CPU with XNNPACK (GPU delegate has compatibility issues)
            options.setUseXNNPACK(true)
            Log.i("SignLanguageAnalyzer", "Using CPU with XNNPACK for TFLite")
            
            tflite = Interpreter(modelBuffer, options)
        } catch (e: Exception) {
            Log.e("SignLanguageAnalyzer", "Error initializing TFLite: ${e.message}")
        }
    }

    private fun loadModelFile(filename: String): ByteBuffer {
        val fileDescriptor = context.assets.openFd(filename)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }

    override fun analyze(imageProxy: ImageProxy) {
        if (handLandmarker == null || tflite == null) {
            imageProxy.close()
            return
        }

        val bitmap = imageProxy.toBitmap()
        val rotationDegrees = imageProxy.imageInfo.rotationDegrees
        val rotatedBitmap = if (rotationDegrees != 0) {
            rotateBitmap(bitmap, rotationDegrees.toFloat())
        } else {
            bitmap
        }

        analyzeFromBitmap(rotatedBitmap)
        imageProxy.close()
    }

    /**
     * Public method to analyze a pre-rotated Bitmap.
     * Used by NativeCameraView's dual-analyzer to share one frame
     * between sign detection and emotion detection.
     */
    fun analyzeFromBitmap(bitmap: Bitmap) {
        if (handLandmarker == null || tflite == null) return

        val mpImage = BitmapImageBuilder(bitmap).build()
        val result = handLandmarker?.detect(mpImage)
        processResult(result)
    }
    
    private fun rotateBitmap(source: Bitmap, angle: Float): Bitmap {
        val matrix = Matrix()
        matrix.postRotate(angle)
        return Bitmap.createBitmap(source, 0, 0, source.width, source.height, matrix, true)
    }

    private fun processResult(result: HandLandmarkerResult?) {
        val landmarks = result?.landmarks()
        
        if (!landmarks.isNullOrEmpty()) {
            // Take the first hand
            val firstHand = landmarks[0]
            
            // Normalize
            val normalized = normalizeLandmarks(firstHand)
            
            // Add to buffer
            updateBuffer(normalized)
            
            // Run inference (it handles padding/throttling internally)
            runInference()
        } else {
             // Handle no hand? Maybe clear buffer or replicate last?
             // For real-time, we typically just wait.
             // Or fill with zeros if we want to reset? 
             // We'll stick to: if no hand, no update to buffer (pause detection).
             // Or strictly, we should probably add a zero-vector to indicate silence.
             // Let's replicate strict behavior: if no hand, maybe add zeros.
             // But the Python code adds zeros: landmarks_data.append([0.0] * 63)
             
             val zeros = FloatArray(NUM_FEATURES) { 0f }
             updateBuffer(zeros)
             
             if (frameBuffer.size == WINDOW_SIZE) {
                 // Optional: Keep forcing inference even if buffer is full of zeros
                 runInference()
             } else {
                 // Also run inference if we are padding
                 runInference()
             }
        }
    }
    
    private fun normalizeLandmarks(landmarks: List<com.google.mediapipe.tasks.components.containers.NormalizedLandmark>): FloatArray {
        // Convert to x,y,z arrays
        // NormalizedLandmark has x,y,z in [0,1].
        // Python code: centered = landmarks - wrist; scale = max(dist).
        
        val flattener = FloatArray(NUM_FEATURES)
        
        // 1. Convert to simple float array of x,y,z
        // AND Find Wrist (index 0)
        val wristX = landmarks[0].x()
        val wristY = landmarks[0].y()
        val wristZ = landmarks[0].z()
        
        // 2. Center and find scale
        var maxDist = 0f
        
        for (i in 0 until 21) {
            val lm = landmarks[i]
            val cx = lm.x() - wristX
            val cy = lm.y() - wristY
            val cz = lm.z() - wristZ
            
            flattener[i * 3] = cx
            flattener[i * 3 + 1] = cy
            flattener[i * 3 + 2] = cz
            
            val dist = sqrt(cx*cx + cy*cy + cz*cz)
            if (dist > maxDist) maxDist = dist
        }
        
        // 3. Scale
        if (maxDist > 0) {
            for (i in 0 until NUM_FEATURES) {
                flattener[i] /= maxDist
            }
        }
        // If maxDist is 0, we return the centered coordinates (which are all 0 if all points are at wrist, or just unscaled)
        // This matches Python: if hand_size > 0: normalized = centered / hand_size else: normalized = centered
        
        return flattener
    }
    
    private fun updateBuffer(newFrame: FloatArray) {
        frameBuffer.add(newFrame)
        if (frameBuffer.size > WINDOW_SIZE) {
            frameBuffer.removeFirst()
        }
    }
    
    private var inferenceCounter = 0
    private val INFERENCE_INTERVAL = 5 // Run every 5 frames for faster response
    private var lastPredictedLabel: String? = null
    private var lastPredictionTime: Long = 0
    
    private fun runInference() {
        if (tflite == null) return
        
        // Throttle inference to avoid spamming results
        inferenceCounter++
        if (inferenceCounter % INFERENCE_INTERVAL != 0) return
        
        // Prepare input: dimensions [1, 60, 63]
        val input = Array(1) { Array(WINDOW_SIZE) { FloatArray(NUM_FEATURES) } }
        
        // Fill input with buffer data, padding with zeros if needed
        // If buffer has N frames (where N < 60), we pad 60-N frames of zeros at the start
        val currentSize = frameBuffer.size
        val paddingNeeded = WINDOW_SIZE - currentSize
        
        // Add padding (zeros)
        for (i in 0 until paddingNeeded) {
             input[0][i] = FloatArray(NUM_FEATURES) { 0f }
        }
        
        // Add actual data
        for (i in 0 until currentSize) {
            input[0][paddingNeeded + i] = frameBuffer[i]
        }
        
        // Prepare output: [1, num_classes]
        val output = Array(1) { FloatArray(labels.size) }
        
        tflite?.run(input, output)
        
        // Parse output - get top 3 for debugging
        val result = output[0]
        val indexedPreds = result.mapIndexed { idx, prob -> idx to prob }
            .sortedByDescending { it.second }
            .take(3)
        
        // Log top 3 predictions for debugging
        val debugStr = indexedPreds.joinToString(", ") { 
            "${labels[it.first]}=${String.format("%.1f", it.second * 100)}%" 
        }
        Log.d("SignLanguageAnalyzer", "Top3: $debugStr")
        
        val maxIndex = indexedPreds[0].first
        val maxScore = indexedPreds[0].second
        
        // Send prediction immediately (no threshold)
         val label = labels[maxIndex]
         val currentTime = SystemClock.uptimeMillis()
         
         // Debounce: only send if different from last prediction or 1 second has passed
         if (label != lastPredictedLabel || currentTime - lastPredictionTime > 1000) {
             onResult(label, maxScore)
             lastPredictedLabel = label
             lastPredictionTime = currentTime
         }
    }
}
