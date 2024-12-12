package com.example.aidemo.fragments

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.os.Bundle
import android.util.Log
import androidx.fragment.app.Fragment
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.ImageView
import android.widget.ProgressBar
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import com.example.aidemo.FaceGroup
import com.example.aidemo.FaceGroupAdapter
import com.example.aidemo.FaceRecognitionMLManager
import com.example.aidemo.FaceRecognitionManager
import com.example.aidemo.R
import kotlinx.coroutines.launch

// TODO: Rename parameter arguments, choose names that match
// the fragment initialization parameters, e.g. ARG_ITEM_NUMBER
private const val ARG_PARAM1 = "param1"
private const val ARG_PARAM2 = "param2"

/**
 * A simple [Fragment] subclass.
 * Use the [FaceRecognizationMLFragments.newInstance] factory method to
 * create an instance of this fragment.
 */
class FaceRecognizationMLFragments : Fragment() {
    // TODO: Rename and change types of parameters
    private var param1: String? = null
    private var param2: String? = null
    private lateinit var faceRecognitionManager: FaceRecognitionMLManager
    private val imageUris = mutableListOf<Uri>()

    // Permission launcher
    private val requestPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()
    ) { permissions ->
        if (permissions.all { it.value }) {
            openGallery()
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


    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        arguments?.let {
            param1 = it.getString(ARG_PARAM1)
            param2 = it.getString(ARG_PARAM2)
        }
    }

    override fun onCreateView(
        inflater: LayoutInflater, container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View? {
        // Inflate the layout for this fragment
        return inflater.inflate(
            R.layout.fragment_face_recognization_m_l_fragments,
            container,
            false
        )
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)
        faceRecognitionManager =
            FaceRecognitionMLManager.getInstance(requireContext().applicationContext)

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
        }
        faceRecognitionManager.processingStatsRef.observe(viewLifecycleOwner){
                it ->
            view.findViewById<ImageView>(R.id.selected_RepresentativeImage).setImageBitmap(faceRecognitionManager.referenceImage)

            view.findViewById<ImageView>(R.id.ref_RepresentativeImage).setImageBitmap(faceRecognitionManager.referenceFace)
        }
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
            openGallery()
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
                faceRecognitionManager.referenceImage =imageBitmap
                view?.findViewById<ImageView>(R.id.selected_RepresentativeImage)?.setImageBitmap(imageBitmap)
                faceRecognitionManager.processReferencePhoto1(imageUris[0])
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
            layoutManager = LinearLayoutManager(context, LinearLayoutManager.HORIZONTAL, false)
            adapter = FaceGroupAdapter(groups,true)
        }
    }

    private fun updateUIWithStats(stats: FaceRecognitionMLManager.ProcessingStats) {
        view?.findViewById<TextView>(R.id.statsTextView)?.text = """
            Total Images: ${stats.totalImages}
            Average Time: ${String.format("%.2f", stats.averageTime)} ms
            Min Time: ${String.format("%.2f", stats.minTime)} ms
            Max Time: ${String.format("%.2f", stats.maxTime)} ms
        """.trimIndent()
    }
}