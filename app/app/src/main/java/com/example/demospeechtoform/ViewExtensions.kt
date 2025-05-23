package com.example.demospeechtoform

import android.graphics.drawable.Drawable
import android.widget.Button
import androidx.annotation.ColorRes
import androidx.core.content.ContextCompat
import androidx.core.graphics.drawable.DrawableCompat

fun Button.setTint(@ColorRes colorResId: Int) {
    // Wrap the drawable for tinting, ensuring it doesn't affect other drawables
    val wrappedDrawable: Drawable = DrawableCompat.wrap(this.background).mutate()
    DrawableCompat.setTint(wrappedDrawable, ContextCompat.getColor(this.context, colorResId))
    this.background = wrappedDrawable
}