package com.example.aidemo

import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.ImageView
import android.widget.TextView
import androidx.recyclerview.widget.RecyclerView
import com.example.aidemo.R

class FaceGroupAdapter(
    private val faceGroups: List<FaceGroup>,
    private val showthreshold:Boolean,
    private val onItemClick: ((FaceGroup) -> Unit)? = null
) : RecyclerView.Adapter<FaceGroupAdapter.FaceGroupViewHolder>() {

    inner class FaceGroupViewHolder(itemView: View) : RecyclerView.ViewHolder(itemView) {
        private val representativeImageView: ImageView = itemView.findViewById(R.id.ivRepresentativeImage)
        private val representativeFaceView: ImageView = itemView.findViewById(R.id.ivRepresentativeFace)
        private val thresholdTextVIew:TextView = itemView.findViewById(R.id.thresholdPer)
        fun bind(faceGroup: FaceGroup) {
            // Set the representative image
            representativeImageView.setImageBitmap(faceGroup.representativeImage)

            // Set the representative face
            representativeFaceView.setImageBitmap(faceGroup.representativeFace)
            thresholdTextVIew.text="${faceGroup.threshold}"

            if(showthreshold){
            thresholdTextVIew.visibility = View.VISIBLE}
            else{
                thresholdTextVIew.visibility = View.GONE
            }

            // Set click listener
            itemView.setOnClickListener {
                onItemClick?.invoke(faceGroup)
            }
        }
    }

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): FaceGroupViewHolder {
        val view = LayoutInflater.from(parent.context)
            .inflate(R.layout.item_face_group, parent, false)
        return FaceGroupViewHolder(view)
    }

    override fun onBindViewHolder(holder: FaceGroupViewHolder, position: Int) {
        holder.bind(faceGroups[position])
    }

    override fun getItemCount(): Int = faceGroups.size
}