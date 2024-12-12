package com.example.aidemo

import android.net.Uri
import android.os.Bundle
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.recyclerview.widget.RecyclerView
import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.widget.ImageView
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.fragment.app.Fragment
import androidx.lifecycle.lifecycleScope
import androidx.recyclerview.widget.LinearLayoutManager
import com.example.aidemo.fragments.FaceRecognitionFragment
import com.example.aidemo.fragments.FaceRecognizationMLFragments
import com.google.android.material.bottomnavigation.BottomNavigationView
import kotlinx.coroutines.launch

class MainActivity : AppCompatActivity() {

    val PERMISSION_REQUEST_CODE = 102
//    private lateinit var faceRecognitionManager: FaceRecognitionManager
    private val imageUris = mutableListOf<Uri>()

    // Permission launcher


    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        replaceFragment(FaceRecognitionFragment())

        // Set up Bottom Navigation
      findViewById<BottomNavigationView>(R.id.bottom_navigation).setOnItemSelectedListener { menuItem ->
            when (menuItem.itemId) {
                R.id.nav_face_recognition -> {
                    replaceFragment(FaceRecognitionFragment())
                    true
                }
                R.id.nav_settings -> {
//
                    replaceFragment(FaceRecognizationMLFragments())
                    true
                }
                else -> false
            }
        }

//        faceRecognitionManager = FaceRecognitionManager.getInstance(applicationContext)
//
//        // Set up observers
//        faceRecognitionManager.isProcessing.observe(this) { isProcessing ->
//            // Update UI to show/hide loading indicator
//            findViewById<ProgressBar>(R.id.progressBar).visibility =
//                if (isProcessing) View.VISIBLE else View.GONE
//        }
//
//        faceRecognitionManager.faceGroups.observe(this) { groups ->
//            // Update UI with face groups
//            updateUIWithFaceGroups(groups)
//        }
//        faceRecognitionManager.processingStatsRef.observe(this){
//            it ->
//            showextrtecFace()
//        }
//
//        faceRecognitionManager.processingStats.observe(this) { stats ->
//            // Update UI with processing stats
//            updateUIWithStats(stats)
//        }
//
//        // Set up button click listener
//        findViewById<Button>(R.id.startButton).setOnClickListener {
//            checkPermissionsAndStartProcess()
////            checkAndRequestPermissions()
//        }
    }
    private fun replaceFragment(fragment: Fragment) {
        supportFragmentManager.beginTransaction()
            .replace(R.id.fragment_container, fragment)
            .commit()
    }
    private fun showextrtecFace(){
//      findViewById<ImageView>(R.id.ref_RepresentativeImage).setImageBitmap(faceRecognitionManager.referenceFace)

    }




//

}