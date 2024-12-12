package com.example.aidemo

// FaceRecognitionManager.kt

import android.content.ContentUris
import android.content.Context
import android.content.res.AssetManager
import android.graphics.*
import android.net.Uri
import android.provider.MediaStore
import android.util.Log
import androidx.exifinterface.media.ExifInterface
import androidx.lifecycle.LiveData
import androidx.lifecycle.MutableLiveData
import com.google.android.gms.tasks.Tasks
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.facedetector.FaceDetector
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.face.Face
import com.google.mlkit.vision.face.FaceDetection
import com.google.mlkit.vision.face.FaceDetectorOptions
import com.google.mlkit.vision.face.FaceLandmark
import kotlinx.coroutines.*
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.io.IOException
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import kotlin.coroutines.resume
import kotlin.coroutines.resumeWithException
import kotlin.coroutines.suspendCoroutine
import kotlin.math.abs
import kotlin.math.atan2

import kotlin.math.sqrt




class FaceRecognitionManager private constructor(private val context: Context) {

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
                    strtgroup()

                }



            }
        } catch (e: Exception) {
            e.printStackTrace()
        }
    }
    suspend fun processReferencePhoto(uri: Uri) = withContext(Dispatchers.Default) {
        try {
            val bitmap = loadAndResizeBitmap(uri)
            val straightenedImage = straightenFace(bitmap)
            val faces = detectFacesMediaPipe(straightenedImage)

            if (faces.isNotEmpty()) {
                // Take the first detected face as reference
                val face = faces[0]
                val croppedFace = cropFace2(straightenedImage, face.boundingBox)
                croppedFace?.let {
                    referenceFace = it
                    referenceEmbedding = generateEmbedding(it)
                    _processingStatsRef.postValue(true)
                    strtgroup()

                }



            }
            else{
                referenceFace = null
                _processingStatsRef.postValue(true)
                _faceGroups.postValue(emptyList())
                _isProcessing.postValue(false)
            }
        } catch (e: Exception) {
            e.printStackTrace()
        }
    }
    suspend fun strtgroup(){

        val photolist = fetchPhotos()
        println("Fethced Phots ${photolist.count()}")

        processGalleryPhotos(photolist)
    }
    fun fetchPhotos(): List<Uri> {
        val photoUris = mutableListOf<Uri>()

        val projection = arrayOf(
            MediaStore.Images.Media._ID,
            MediaStore.Images.Media.DATE_MODIFIED
        )

        val selection = "${MediaStore.Images.Media.BUCKET_DISPLAY_NAME} = ?"
        val selectionArgs = arrayOf("Camera")

        // Remove LIMIT from the sortOrder and handle it in code
        val sortOrder = "${MediaStore.Images.Media.DATE_MODIFIED} DESC"

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
    suspend fun processGalleryPhotos(imageUris: List<Uri>) = withContext(Dispatchers.Default) {
        _isProcessing.postValue(true)
        val faceGroups = mutableListOf<FaceGroup>()
        println("Process photos ${imageUris.count()  } $imageUris")

        imageUris.forEach { uri ->
            val startTime = System.currentTimeMillis()
            try {
                val bitmap = loadAndResizeBitmap(uri)
                val straightenedImage =correctImageOrientation(bitmap)
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

        // First, get the image orientation from EXIF metadata
        val orientation = context.contentResolver.openInputStream(uri)?.use { inputStream ->
            val exif = ExifInterface(inputStream)
            exif.getAttributeInt(
                ExifInterface.TAG_ORIENTATION,
                ExifInterface.ORIENTATION_NORMAL
            )
        } ?: ExifInterface.ORIENTATION_NORMAL

        // Calculate sample size for resizing
        context.contentResolver.openInputStream(uri)?.use {
            BitmapFactory.decodeStream(it, null, options)
        }

        options.apply {
            inJustDecodeBounds = false
            inSampleSize = calculateInSampleSize(this, 1024, 1024)
        }

        // Load the bitmap
        val bitmap = context.contentResolver.openInputStream(uri)?.use {
            BitmapFactory.decodeStream(it, null, options)
        } ?: throw IllegalStateException("Could not load bitmap")

        // Rotate the bitmap if needed
        return when (orientation) {
            ExifInterface.ORIENTATION_ROTATE_90 -> rotateBitmap(bitmap, 90f)
            ExifInterface.ORIENTATION_ROTATE_180 -> rotateBitmap(bitmap, 180f)
            ExifInterface.ORIENTATION_ROTATE_270 -> rotateBitmap(bitmap, 270f)
            else -> bitmap
        }
    }

    // Helper function to rotate bitmap
    private fun rotateBitmap(bitmap: Bitmap, degrees: Float): Bitmap {
        val matrix = Matrix().apply { postRotate(degrees) }
        return Bitmap.createBitmap(
            bitmap,
            0, 0,
            bitmap.width,
            bitmap.height,
            matrix,
            true
        )
    }
    private fun loadAndResizeBitmap1(uri: Uri): Bitmap {
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
    private fun straightenFace(bitmap: Bitmap): Bitmap {
        // Create a matrix for rotation
        val matrix = Matrix()

        // Detect face landmarks to determine rotation angle
        val options = FaceDetectorOptions.Builder()
            .setLandmarkMode(FaceDetectorOptions.LANDMARK_MODE_ALL)
            .build()

        val detector = FaceDetection.getClient(options)
        val image = InputImage.fromBitmap(bitmap, 0)

        return try {
            // Use synchronous wait to get face detection results
            val faces = Tasks.await(detector.process(image))

            if (faces.isNotEmpty()) {
                val face = faces.first()
                val landmarks = face.allLandmarks

                // Find left and right eye landmarks
                val leftEye = landmarks.find { it.landmarkType == FaceLandmark.LEFT_EYE }
                val rightEye = landmarks.find { it.landmarkType == FaceLandmark.RIGHT_EYE }

                if (leftEye != null && rightEye != null) {
                    // Calculate rotation angle
                    val angleRadians = atan2(
                        rightEye.position.y - leftEye.position.y,
                        rightEye.position.x - leftEye.position.x
                    )
                    val angleDegrees = Math.toDegrees(angleRadians.toDouble()).toFloat()
                    Log.d("ImageRotation","angle degree $angleDegrees  $angleRadians")

                    // Determine if rotation is significant (> 5 degrees)
                    if (abs(angleDegrees) >45) {
                        // Rotate the bitmap
                        matrix.postRotate(-angleDegrees)
                        Bitmap.createBitmap(
                            bitmap,
                            0, 0,
                            bitmap.width, bitmap.height,
                            matrix,
                            true
                        )
                    } else {
                        // Return original bitmap if rotation is minimal
                        bitmap
                    }
                } else {
                    // No eye landmarks found, return original bitmap
                    bitmap
                }
            } else {
                // No faces detected, return original bitmap
                bitmap
            }
        } catch (e: Exception) {
            // If any error occurs during face detection or rotation, return original bitmap
            println("Error in face straightening: ${e.message}")
            bitmap
        }
    }
    private fun correctImageOrientation(bitmap: Bitmap): Bitmap {
        // Detect face landmarks to determine rotation
        val options = FaceDetectorOptions.Builder()
            .setLandmarkMode(FaceDetectorOptions.LANDMARK_MODE_ALL)
            .build()

        val detector = FaceDetection.getClient(options)
        val image = InputImage.fromBitmap(bitmap, 0)

        try {
            // Synchronous face detection
            val faces = Tasks.await(detector.process(image))

            if (faces.isNotEmpty()) {
                val face = faces.first()
                val landmarks = face.allLandmarks

                // Find left and right eye landmarks
                val leftEye = landmarks.find { it.landmarkType == FaceLandmark.LEFT_EYE }
                val rightEye = landmarks.find { it.landmarkType == FaceLandmark.RIGHT_EYE }

                if (leftEye != null && rightEye != null) {
                    // Determine eye alignment
                    val isVerticalAlignment = isEyesVerticallyAligned(leftEye, rightEye)

                    // Debug prints
                    println("Left Eye: (${leftEye.position.x}, ${leftEye.position.y})")
                    println("Right Eye: (${rightEye.position.x}, ${rightEye.position.y})")
                    println("Is Vertically Aligned: $isVerticalAlignment")

                    val matrix = Matrix()

                    // Rotation logic based on vertical alignment
                    if (isVerticalAlignment) {
                        // Rotate to make eyes horizontal
                        matrix.postRotate(90f)
                        return Bitmap.createBitmap(
                            bitmap,
                            0, 0,
                            bitmap.width,
                            bitmap.height,
                            matrix,
                            true
                        )
                    }
                }
            }
        } catch (e: Exception) {
            println("Face detection error: ${e.message}")
        }

        // Return original bitmap if no rotation needed
        return bitmap
    }

    // Function to check if eyes are vertically aligned
    private fun isEyesVerticallyAligned(leftEye: FaceLandmark, rightEye: FaceLandmark): Boolean {
        // Calculate the difference in x and y coordinates
        val xDifference = abs(leftEye.position.x - rightEye.position.x)
        val yDifference = abs(leftEye.position.y - rightEye.position.y)
        println("Is Vertically Aligned: $xDifference  $yDifference")
        // Determine alignment based on relative differences
        // If y-difference is significantly larger than x-difference, eyes are vertically aligned
        val alignmentRatio = yDifference / (xDifference + 1f)  // Add 1 to prevent division by zero

        // Adjust this threshold based on your specific use case
        return alignmentRatio > 1.5  // Means y-difference is 1.5 times more than x-difference
    }

    private fun straightenFace1(bitmap: Bitmap): Bitmap {
        // Detect face landmarks to determine rotation
        val options = FaceDetectorOptions.Builder()
            .setLandmarkMode(FaceDetectorOptions.LANDMARK_MODE_ALL)
            .build()

        val detector = FaceDetection.getClient(options)
        val image = InputImage.fromBitmap(bitmap, 0)

        try {
            // Synchronous face detection
            val faces = Tasks.await(detector.process(image))

            if (faces.isNotEmpty()) {
                val face = faces.first()
                val landmarks = face.allLandmarks

                // Find left and right eye landmarks
                val leftEye = landmarks.find { it.landmarkType == FaceLandmark.LEFT_EYE }
                val rightEye = landmarks.find { it.landmarkType == FaceLandmark.RIGHT_EYE }

                if (leftEye != null && rightEye != null) {
                    // Calculate angle between eyes
                    val angleRadians = atan2(
                        rightEye.position.y - leftEye.position.y,
                        rightEye.position.x - leftEye.position.x
                    )
                    val angleDegrees = Math.toDegrees(angleRadians.toDouble()).toFloat()
                    Log.d("ImageRotation","angle degree $angleDegrees  $angleRadians")
                    // Detailed rotation logic
                    val matrix = Matrix()
                    when {
                        // Tilted 90 degrees clockwise (eyes are vertical)
                        angleDegrees > 60 -> {
                            Log.d("ImageRotation","rotating 90 degree clockwise")
                            matrix.postRotate(-90f)
                            return Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
                        }
                        // Tilted 90 degrees counter-clockwise (eyes are vertical)
                        angleDegrees < -60 -> {
                            Log.d("ImageRotation","rotating 90 degree anit clockwise")
                            matrix.postRotate(90f)
                            return Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
                        }
                        // Upside down (nearly horizontal eyes)
                        abs(angleDegrees) > 150 -> {
                            Log.d("ImageRotation","rotating 180 degree colkwise")
                            matrix.postRotate(180f)
                            return Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
                        }
                    }
                }
            }
        } catch (e: Exception) {
            println("Face detection error: ${e.message}")
        }

        // Return original bitmap if no rotation needed
        return bitmap
    }
    private fun straightenFace11(bitmap: Bitmap): Bitmap {

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
    data class Face1(
        val boundingBox: RectF,
        val confidence: Float
        // Add other properties you want to track
    )
    fun cropFace2(bitmap: Bitmap, boundingBox: RectF): Bitmap? {
        return try {
            Bitmap.createBitmap(
                bitmap,
                boundingBox.left.toInt(),
                boundingBox.top.toInt(),
                boundingBox.width().toInt(),
                boundingBox.height().toInt()
            )
        } catch (e: Exception) {
            null
        }
    }



    private suspend fun detectFacesMediaPipe(bitmap: Bitmap): List<Face1> = suspendCancellableCoroutine { continuation ->
        try {
            // Set up base options
            val baseOptionsBuilder = BaseOptions.builder()

                .setModelAssetPath("blaze_face_short_range.tflite")


            // Create detector options
            val options = FaceDetector.FaceDetectorOptions.builder()
                .setBaseOptions(baseOptionsBuilder.build())
                .setMinDetectionConfidence(0.3f)
                .setRunningMode(RunningMode.IMAGE)

                .build()

            // Create the face detector
            val faceDetector = FaceDetector.createFromOptions(context, options)

            // Convert bitmap to MediaPipe image
            val mpImage = BitmapImageBuilder(bitmap).build()

            // Detect faces
            val faceDetectorResult = faceDetector.detect(mpImage)

            // Convert MediaPipe detections to your Face type
            val faces = faceDetectorResult.detections().map { detection ->
                Face1(
                    boundingBox = detection.boundingBox(),
                    confidence = detection.categories()[0].score()
                )
            }

            // Close the detector
            faceDetector.close()

            // Resume the coroutine with detected faces
            continuation.resume(faces)
        } catch (e: IllegalStateException) {
            println("Face detector failed to initialize: ${e.message}")
            continuation.resumeWithException(e)
        } catch (e: RuntimeException) {
            println("Face detector failed to process: ${e.message}")
            continuation.resumeWithException(e)
        } catch (e: Exception) {
            println("Face detection error: ${e.message}")
            continuation.resumeWithException(e)
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
        faceGroups.add(
            FaceGroup(
                representativeImage = originalImage,
                images = mutableListOf(originalImage),
                representativeFace = croppedFace,
                threshold =0.0f
            )
        )
//        var similarity  = referenceFace?.let { compareFaces1(croppedFace, it) }
//        if(similarity!! > 0.70f){
////            println("ML found Similar Photos $similarity")
//            Log.d("ML","ML found Similar Photos $similarity")
//            faceGroups.add(
//                FaceGroup(
//                    representativeImage = originalImage,
//                    images = mutableListOf(originalImage),
//                    representativeFace = croppedFace,
//                    threshold = similarity
//                )
//            )
//        }

//        if (!matched) {
//        println("Found Similar $referenceFace")
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
//        else{
//            println("NOt Found Simlier Photos")
//        }
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
        private var instance: FaceRecognitionManager? = null
        private const val FETCH_LIMIT = 30
        fun getInstance(context: Context): FaceRecognitionManager {
            return instance ?: synchronized(this) {
                instance ?: FaceRecognitionManager(context).also { instance = it }
            }
        }
    }
}