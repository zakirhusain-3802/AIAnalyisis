package com.example.aidemo

// FaceRecognitionManager.kt

import android.content.ContentUris
import android.content.Context
import android.content.res.AssetManager
import android.graphics.*
import android.net.Uri
import android.provider.MediaStore
import android.util.Log
import androidx.lifecycle.LiveData
import androidx.lifecycle.MutableLiveData
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.facedetector.FaceDetector
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.face.Face
import com.google.mlkit.vision.face.FaceDetection
import com.google.mlkit.vision.face.FaceDetectorOptions
import kotlinx.coroutines.*
import org.tensorflow.lite.Interpreter
import java.io.File
import java.io.FileInputStream
import java.io.IOException
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import kotlin.coroutines.resume
import kotlin.coroutines.resumeWithException
import kotlin.coroutines.suspendCoroutine
import kotlin.math.sqrt




class FaceRecognitionMLManager private constructor(private val context: Context) {

    private val MODEL_FILE = "facenet_512_int_quantized.tflite"

    private var interpreter: Interpreter? = null
    private val processingTimes = mutableListOf<Long>()
    private var totalProcessingTime: Long = 0
    private var imageProcessedCount: Int = 0

    private val _isProcessing = MutableLiveData<Boolean>()
    val isProcessing: LiveData<Boolean> = _isProcessing

    private val _faceGroups = MutableLiveData<List<FaceGroup>>()
    val faceGroups: LiveData<List<FaceGroup>> = _faceGroups

    private val _processingStats = MutableLiveData<ProcessingStats>()
    val processingStats: LiveData<ProcessingStats> = _processingStats

    private val _processingStatsRef = MutableLiveData<Boolean>()
    val processingStatsRef: LiveData<Boolean> = _processingStatsRef

    data class ProcessingStats(
        val totalImages: Int = 0,
        val averageTime: Double = 0.0,
        val minTime: Double = 0.0,
        val maxTime: Double = 0.0,
        val totalTime: Double = 0.0,
        val medianTime: Double = 0.0,
        val fastestQuartile: Double = 0.0,
        val slowestQuartile: Double = 0.0
    )



    init {
        setupFaceNetModel1()
    }

    @Throws(IOException::class)
    fun loadModelFile(assetManager: AssetManager, modelPath: String?): MappedByteBuffer {
        val fileDescriptor = assetManager.openFd(modelPath!!)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }
    private fun setupFaceNetModel1() {
        try {
//
            val options = Interpreter.Options().apply {
                numThreads = 4
                // Add error delegate to help with debugging
                setUseXNNPACK(true)
            }


            interpreter = Interpreter(loadModelFile(context.assets,MODEL_FILE), options)

            // Debug model input/output details
            val inputs = interpreter?.getInputTensor(0)
            val outputs = interpreter?.getOutputTensor(0)
            Log.d("Similar", """
                Model details:
                Input shape: ${inputs?.shape()?.joinToString()}
                Input dataType: ${inputs?.dataType()}
                Output shape: ${outputs?.shape()?.joinToString()}
                Output dataType: ${outputs?.dataType()}
            """.trimIndent())

        } catch (e: Exception) {
            Log.e("Similar", "Error setting up FaceNet model", e)
            e.printStackTrace()
        }
    }
    private var referenceEmbedding: FloatArray? = null
    var referenceFace: Bitmap? = null
    var referenceImage: Bitmap? = null

    private val _matchedFaces = MutableLiveData<List<Bitmap>>()
    val matchedFaces: LiveData<List<Bitmap>> = _matchedFaces



    suspend fun processReferencePhoto1(uri: Uri) = withContext(Dispatchers.Default) {
        try {
            val bitmap = loadAndResizeBitmap(uri)
            val straightenedImage = straightenFace(bitmap)
            val faces = detectFaces(straightenedImage)

            if (faces.isNotEmpty()) {
                // Take the first detected face as reference
                val face = faces[0]
                val croppedFace = cropFace(straightenedImage, face.boundingBox)
                croppedFace?.let {
                    referenceFace = it
                    referenceEmbedding = generateEmbedding(it)
                    _processingStatsRef.postValue(true)
                    strtgroup1()

                }



            }
        } catch (e: Exception) {
            e.printStackTrace()
        }
    }

    suspend fun strtgroup1(){

        val photolist = fetchPhotos()
        println("Fethced Phots ${photolist.count()}")

        processGalleryPhotos1(photolist)
    }
    fun fetchPhotos(): List<Uri> {
        val photoUris = mutableListOf<Uri>()

        val projection = arrayOf(
            MediaStore.Images.Media._ID,
            MediaStore.Images.Media.DATE_TAKEN
        )

        val selection = "${MediaStore.Images.Media.BUCKET_DISPLAY_NAME} = ?"
        val selectionArgs = arrayOf("Camera")

        // Remove LIMIT from the sortOrder and handle it in code
        val sortOrder = "${MediaStore.Images.Media.DATE_TAKEN} DESC"

        context.contentResolver.query(
            MediaStore.Images.Media.EXTERNAL_CONTENT_URI,
            projection,
            selection,
            selectionArgs,
            sortOrder
        )?.use { cursor ->
            val idColumn = cursor.getColumnIndexOrThrow(MediaStore.Images.Media._ID)

            var count = 0
            while (cursor.moveToNext() && count < FETCH_LIMIT) {
                val id = cursor.getLong(idColumn)
                val contentUri = ContentUris.withAppendedId(
                    MediaStore.Images.Media.EXTERNAL_CONTENT_URI,
                    id
                )
                photoUris.add(contentUri)
                count++
            }
        }

        return photoUris
    }


    suspend fun processGalleryPhotos1(imageUris: List<Uri>) = withContext(Dispatchers.Default) {
        _isProcessing.postValue(true)
        val faceGroups = mutableListOf<FaceGroup>()
        println("Process photos ${imageUris.count()  } $imageUris")

        imageUris.forEach { uri ->
            val startTime = System.currentTimeMillis()
            try {
                val bitmap = loadAndResizeBitmap(uri)
                val straightenedImage = straightenFace(bitmap)
                detectFaces(straightenedImage)?.forEach { face ->
                    val croppedFace = cropFace(straightenedImage, face.boundingBox)
                    croppedFace?.let { processDetectedFace(it, straightenedImage, faceGroups)

                    println("Faces Found  ")
                    }
                }

                val processingTime = System.currentTimeMillis() - startTime
                processingTimes.add(processingTime)
                totalProcessingTime += processingTime
                imageProcessedCount++

            } catch (e: Exception) {
                println("Error in prcessing $e")
                e.printStackTrace()
            }
        }

        updateProcessingStats()
        _faceGroups.postValue(faceGroups)
        _isProcessing.postValue(false)
    }

    private fun loadAndResizeBitmap(uri: Uri): Bitmap {
        val options = BitmapFactory.Options().apply {
            inJustDecodeBounds = true
        }
        context.contentResolver.openInputStream(uri)?.use {
            BitmapFactory.decodeStream(it, null, options)
        }

        options.apply {
            inJustDecodeBounds = false
            inSampleSize = calculateInSampleSize(this, 1024, 1024)
        }

        return context.contentResolver.openInputStream(uri)?.use {
            BitmapFactory.decodeStream(it, null, options)
        } ?: throw IllegalStateException("Could not load bitmap")
    }
    private fun calculateInSampleSize(options: BitmapFactory.Options, reqWidth: Int, reqHeight: Int): Int {
        // Get the dimensions of the bitmap
        val (height: Int, width: Int) = options.run { outHeight to outWidth }
        var inSampleSize = 1

        if (height > reqHeight || width > reqWidth) {
            val halfHeight: Int = height / 2
            val halfWidth: Int = width / 2

            // Calculate the largest inSampleSize value that is a power of 2 and keeps both
            // height and width larger than the requested height and width.
            while (halfHeight / inSampleSize >= reqHeight && halfWidth / inSampleSize >= reqWidth) {
                inSampleSize *= 2
            }
        }

        return inSampleSize
    }


    private suspend fun detectFaces(bitmap: Bitmap) = suspendCoroutine<List<Face>> { continuation ->
        val options = FaceDetectorOptions.Builder()
            .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_ACCURATE)
            .setLandmarkMode(FaceDetectorOptions.LANDMARK_MODE_ALL)
            .build()

        val detector = FaceDetection.getClient(options)
        val image = InputImage.fromBitmap(bitmap, 0)



        detector.process(image)
            .addOnSuccessListener { faces ->
                continuation.resume(faces)
            }
            .addOnFailureListener { e ->
                println("Face not found $e")
                continuation.resumeWithException(e)
            }
    }
    data class Face1(
        val boundingBox: RectF,
        val confidence: Float
        // Add other properties you want to track
    )


    private fun straightenFace(bitmap: Bitmap): Bitmap {
        // Implementation similar to iOS version's straightenAndDetectFace
        // For brevity, returning original bitmap. Implement rotation logic based on eye landmarks if needed
        return bitmap
    }

    private fun cropFace(bitmap: Bitmap, boundingBox: Rect): Bitmap? {
        return try {
            Bitmap.createBitmap(
                bitmap,
                boundingBox.left,
                boundingBox.top,
                boundingBox.width(),
                boundingBox.height()
            )
        } catch (e: Exception) {
            print("Execption in crop $e")
            null
        }
    }

    private fun processDetectedFace(
        croppedFace: Bitmap,
        originalImage: Bitmap,
        faceGroups: MutableList<FaceGroup>
    ) {
        var matched = false
//        for (group in faceGroups) {
//            if (compareFaces(croppedFace, group.representativeFace)) {
//                group.images.add(originalImage)
//                matched = true
//                break
//            }
//        }
//
//        if (!matched) {
//        println("Found Similar $referenceFace")
        var similarity  = referenceFace?.let { compareFaces1(croppedFace, it) }
        if(similarity!! > 0.70f){
//            println("ML found Similar Photos $similarity")
            Log.d("ML","ML found Similar Photos $similarity")
            faceGroups.add(
                FaceGroup(
                    representativeImage = originalImage,
                    images = mutableListOf(originalImage),
                    representativeFace = croppedFace,
                    threshold = similarity
                )
            )
        }
//        if(referenceFace?.let { compareFaces(croppedFace, it) }!!){
//            println("Found Simlier Photos1")
//            faceGroups.add(
//                FaceGroup(
//                    representativeImage = originalImage,
//                    images = mutableListOf(originalImage),
//                    representativeFace = croppedFace,
//                    threshold = 0.0F
//                )
//            )
////        }
//        }
        else{
            println("NOt Found Simlier Photos")
        }
    }
    private fun compareFaces(face1: Bitmap, face2: Bitmap, threshold: Float = 0.65f): Boolean {
        val embedding1 = generateEmbedding(face1)
        val embedding2 = generateEmbedding(face2)

//        // Debug embeddings
//        Log.d("Similar", "Embedding 1 first 10 values: ${embedding1?.take(10)?.joinToString()}")
//        Log.d("Similar", "Embedding 2 first 10 values: ${embedding2?.take(10)?.joinToString()}")

        return if (embedding1 != null && embedding2 != null) {
            val similarity = cosineSimilarity(embedding1, embedding2)
            Log.d("Similar", "Raw similarity score: $similarity")

            // Debug intermediate calculations
            val dotProduct = embedding1.zip(embedding2).sumOf { (a, b) -> (a * b).toDouble() }
            val norm1 = sqrt(embedding1.sumOf { it * it.toDouble() })
            val norm2 = sqrt(embedding2.sumOf { it * it.toDouble() })

//            Log.d("Similar", """
//                Debugging similarity calculation:
//                - Dot product: $dotProduct
//                - Norm1: $norm1
//                - Norm2: $norm2
//            """.trimIndent())

            similarity > threshold
        } else {
            Log.e("Similar", "One or both embeddings are null")
            false
        }
    }
    private fun compareFaces1(face1: Bitmap, face2: Bitmap, threshold: Float = 0.65f): Float {
        val embedding1 = generateEmbedding(face1)
        val embedding2 = generateEmbedding(face2)

//        // Debug embeddings
//        Log.d("Similar", "Embedding 1 first 10 values: ${embedding1?.take(10)?.joinToString()}")
//        Log.d("Similar", "Embedding 2 first 10 values: ${embedding2?.take(10)?.joinToString()}")

        if (embedding1 != null && embedding2 != null) {
            val similarity = cosineSimilarity(embedding1, embedding2)
            Log.d("Similar", "Raw similarity score: $similarity")

            // Debug intermediate calculations
            val dotProduct = embedding1.zip(embedding2).sumOf { (a, b) -> (a * b).toDouble() }
            val norm1 = sqrt(embedding1.sumOf { it * it.toDouble() })
            val norm2 = sqrt(embedding2.sumOf { it * it.toDouble() })

//            Log.d("Similar", """
//                Debugging similarity calculation:
//                - Dot product: $dotProduct
//                - Norm1: $norm1
//                - Norm2: $norm2
//            """.trimIndent())

          return  similarity
        } else {
            Log.e("Similar", "One or both embeddings are null")
            return  0.0f
        }
    }

    private fun generateEmbedding(face: Bitmap): FloatArray? {
        val outputArray = Array(1) { FloatArray(512) } // Changed to match model output shape [1, 512]

        try {
            val inputArray = preprocessImage(face)
            inputArray.rewind()

            interpreter?.run(inputArray, outputArray)

            // Return the first (and only) array from the 2D array
            return outputArray[0]
        } catch (e: Exception) {
            Log.e("Similar", "Error generating embedding", e)
            return null
        }
    }
    private fun preprocessImage(bitmap: Bitmap): ByteBuffer {
        val inputSize = 160
        val resizedBitmap = Bitmap.createScaledBitmap(bitmap, inputSize, inputSize, true)
        val byteBuffer = ByteBuffer.allocateDirect(inputSize * inputSize * 3 * 4)
        byteBuffer.order(ByteOrder.nativeOrder())

        val intValues = IntArray(inputSize * inputSize)
        resizedBitmap.getPixels(intValues, 0, inputSize, 0, 0, inputSize, inputSize)

        // Debug pixel values
        var maxVal = Float.MIN_VALUE
        var minVal = Float.MAX_VALUE

        intValues.forEach { pixelValue ->
            val r = ((pixelValue shr 16) and 0xFF)
            val g = ((pixelValue shr 8) and 0xFF)
            val b = (pixelValue and 0xFF)

            // Normalize to [-1, 1] instead of [0, 1]
            byteBuffer.putFloat((r - 127.5f) / 127.5f)
            byteBuffer.putFloat((g - 127.5f) / 127.5f)
            byteBuffer.putFloat((b - 127.5f) / 127.5f)

            maxVal = maxOf(maxVal, r.toFloat(), g.toFloat(), b.toFloat())
            minVal = minOf(minVal, r.toFloat(), g.toFloat(), b.toFloat())
        }

        byteBuffer.rewind()
        Log.d("Similar", """
            Image preprocessing stats:
            - Max pixel value: $maxVal
            - Min pixel value: $minVal
            - Buffer capacity: ${byteBuffer.capacity()}
            - Expected capacity: ${inputSize * inputSize * 3 * 4}
        """.trimIndent())
        return byteBuffer
    }

//

    private fun cosineSimilarity(embedding1: FloatArray, embedding2: FloatArray): Float {
        var dotProduct = 0f
        var norm1 = 0f
        var norm2 = 0f

        for (i in embedding1.indices) {
            dotProduct += embedding1[i] * embedding2[i]
            norm1 += embedding1[i] * embedding1[i]
            norm2 += embedding2[i] * embedding2[i]
        }

        norm1 = sqrt(norm1)
        norm2 = sqrt(norm2)

        return if (norm1 > 0 && norm2 > 0) {
            dotProduct / (norm1 * norm2)
        } else 0f
    }

    private fun updateProcessingStats() {
        val sortedTimes = processingTimes.sorted()
        val stats = ProcessingStats(
            totalImages = processingTimes.size,
            averageTime = totalProcessingTime.toDouble() / processingTimes.size,
            minTime = sortedTimes.first().toDouble(),
            maxTime = sortedTimes.last().toDouble(),
            totalTime = totalProcessingTime.toDouble(),
            medianTime = sortedTimes[sortedTimes.size / 2].toDouble(),
            fastestQuartile = sortedTimes[(sortedTimes.size * 0.25).toInt()].toDouble(),
            slowestQuartile = sortedTimes[(sortedTimes.size * 0.75).toInt()].toDouble()
        )
        _processingStats.postValue(stats)
    }

    companion object {
        @Volatile
        private var instance: FaceRecognitionMLManager? = null
        private const val FETCH_LIMIT = 30
        fun getInstance(context: Context): FaceRecognitionMLManager {
            return instance ?: synchronized(this) {
                instance ?: FaceRecognitionMLManager(context).also { instance = it }
            }
        }
    }
}
data class FaceGroup(
    val representativeImage: Bitmap,
    val images: MutableList<Bitmap>,
    val representativeFace: Bitmap,
    val threshold: Float
)