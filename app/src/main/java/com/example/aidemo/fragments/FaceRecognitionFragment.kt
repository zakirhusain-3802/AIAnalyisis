package com.example.aidemo.fragments

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.os.Bundle
import android.util.Log
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.ImageView
import android.widget.ProgressBar
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.core.content.ContextCompat
import androidx.fragment.app.Fragment
import androidx.lifecycle.lifecycleScope
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import com.example.aidemo.FaceGroup
import com.example.aidemo.FaceGroupAdapter
import com.example.aidemo.FaceRecognitionManager
import com.example.aidemo.R
import kotlinx.coroutines.launch

class FaceRecognitionFragment : Fragment() {
    private lateinit var faceRecognitionManager: FaceRecognitionManager
    private val imageUris = mutableListOf<Uri>()

    // Permission launcher
    private val requestPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()
    ) { permissions ->
        if (permissions.all { it.value }) {
//            openGallery()
            proceesPhotos()
        } else {
            Toast.makeText(requireContext(), "Permissions required to access photos", Toast.LENGTH_LONG).show()
        }
    }

    // Gallery launcher
    private val galleryLauncher = registerForActivityResult(
        ActivityResultContracts.GetMultipleContents()
    ) { uris ->
        if (uris.isNotEmpty()) {
            imageUris.clear()
            imageUris.addAll(uris)
            startFaceRecognition()
        }
    }

    override fun onCreateView(
        inflater: LayoutInflater, container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View {
        return inflater.inflate(R.layout.fragment_face_recognition, container, false)
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)

        faceRecognitionManager =
            FaceRecognitionManager.getInstance(requireContext().applicationContext)

        // Set up observers
        faceRecognitionManager.isProcessing.observe(viewLifecycleOwner) { isProcessing ->
            view.findViewById<ProgressBar>(R.id.progressBar).visibility =
                if (isProcessing) View.VISIBLE else View.GONE
        }

        faceRecognitionManager.faceGroups.observe(viewLifecycleOwner) { groups ->
            updateUIWithFaceGroups(groups)
        }

        faceRecognitionManager.processingStats.observe(viewLifecycleOwner) { stats ->
            updateUIWithStats(stats)
        }

        view.findViewById<View>(R.id.startButton).setOnClickListener {
            checkPermissionsAndStartProcess()
//            proceesPhotos()
        }
        faceRecognitionManager.processingStatsRef.observe(viewLifecycleOwner){
            it ->
            showextrtecFace()
            view.findViewById<ImageView>(R.id.selected_RepresentativeImage).setImageBitmap(faceRecognitionManager.referenceImage)
            view.findViewById<ImageView>(R.id.ref_RepresentativeImage).setImageBitmap(faceRecognitionManager.referenceFace)
        }
    }

    private fun proceesPhotos() {
        lifecycleScope.launch {
            faceRecognitionManager.strtgroup()}
    }

    private fun showextrtecFace(){

    }

    private fun checkPermissionsAndStartProcess() {
        val permissions = if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.TIRAMISU) {
            arrayOf(
                Manifest.permission.CAMERA,
                Manifest.permission.READ_MEDIA_IMAGES
            )
        } else {
            arrayOf(
                Manifest.permission.CAMERA,
                Manifest.permission.READ_EXTERNAL_STORAGE

            )
        }

        if (permissions.all {
                context?.let { it1 -> ContextCompat.checkSelfPermission(it1, it) } == PackageManager.PERMISSION_GRANTED
            }) {
//            openGallery()
            proceesPhotos()
        } else {
            requestPermissionLauncher.launch(permissions)
        }
    }

    private fun openGallery() {
        galleryLauncher.launch("image/*")
    }

    private fun startFaceRecognition() {
        lifecycleScope.launch {
            try {
                val imageBitmap = getBitmapFromUri(requireContext(), imageUris[0])
                faceRecognitionManager.referenceImage = imageBitmap
                view?.findViewById<ImageView>(R.id.selected_RepresentativeImage)?.setImageBitmap(imageBitmap)
                faceRecognitionManager.processReferencePhoto(imageUris[0])
            } catch (e: Exception) {
                Toast.makeText(requireContext(), "Error processing images: ${e.message}", Toast.LENGTH_LONG).show()
            }
        }
    }

    private fun getBitmapFromUri(context: Context, uri: Uri): Bitmap? {
        return try {
            val inputStream = context.contentResolver.openInputStream(uri)
            BitmapFactory.decodeStream(inputStream).also {
                inputStream?.close()
            }
        } catch (e: Exception) {
            e.printStackTrace()
            null
        }
    }

    private fun updateUIWithFaceGroups(groups: List<FaceGroup>) {
        Log.d("Update", "Update Face Grouup")
        view?.findViewById<RecyclerView>(R.id.recyclerView)?.apply {
            layoutManager = LinearLayoutManager(context, LinearLayoutManager.VERTICAL, false)
            adapter = FaceGroupAdapter(groups,false)
        }
    }

    private fun updateUIWithStats(stats: FaceRecognitionManager.ProcessingStats) {
        view?.findViewById<TextView>(R.id.statsTextView)?.text = """
            Total Images: ${stats.totalImages}
            Average Time: ${String.format("%.2f", stats.averageTime)} ms
            Min Time: ${String.format("%.2f", stats.minTime)} ms
            Max Time: ${String.format("%.2f", stats.maxTime)} ms
        """.trimIndent()
    }
}
