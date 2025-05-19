import java.util.Properties

plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.android)
    alias(libs.plugins.kotlin.compose)
    alias(libs.plugins.kotlin.devtoolsKsp)
}

val localProperties = Properties()
val localPropertiesFile = rootProject.file("local.properties")
if (localPropertiesFile.exists()) {
    localPropertiesFile.inputStream().use { input ->
        localProperties.load(input)
    }
}

android {
    namespace = "com.example.demospeechtoform"
    compileSdk = 35

    defaultConfig {
        applicationId = "com.example.demospeechtoform"
        minSdk = 34
        targetSdk = 35
        versionCode = 1
        versionName = "1.0"

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"

        // Make API Key and Base URL available in BuildConfig
        val apiKey = localProperties.getProperty("API_KEY", "")
        val baseUrl = localProperties.getProperty("BASE_URL", "")
        val clientTenant = localProperties.getProperty("CLIENT_TENANT", "")
        val clientApplication = localProperties.getProperty("CLIENT_APPLICATION", "")

        // Add as BuildConfig fields
        buildConfigField("String", "API_KEY", "\"$apiKey\"")
        buildConfigField("String", "BASE_URL", "\"$baseUrl\"")
        buildConfigField("String", "CLIENT_TENANT", "\"$clientTenant\"")
        buildConfigField("String", "CLIENT_APPLICATION", "\"$clientApplication\"")
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_11
        targetCompatibility = JavaVersion.VERSION_11
    }
    kotlinOptions {
        jvmTarget = "11"
    }
    buildFeatures {
        compose = true
        viewBinding = true
        buildConfig = true
    }
}

dependencies {

    implementation(libs.androidx.core.ktx)
    implementation(libs.androidx.lifecycle.runtime.ktx)
    implementation(libs.androidx.activity.compose)
    implementation(platform(libs.androidx.compose.bom))
    implementation(libs.androidx.ui)
    implementation(libs.androidx.ui.graphics)
    implementation(libs.androidx.ui.tooling.preview)
    implementation(libs.androidx.material3)
    implementation(libs.androidx.constraintlayout)
    implementation(libs.material)
    implementation(libs.androidx.appcompat)
    implementation(libs.androidx.navigation.fragment.ktx)
    implementation(libs.androidx.navigation.ui.ktx)
    implementation(libs.androidx.activity)

    testImplementation(libs.junit)
    androidTestImplementation(libs.androidx.junit)
    androidTestImplementation(libs.androidx.espresso.core)
    androidTestImplementation(platform(libs.androidx.compose.bom))
    androidTestImplementation(libs.androidx.ui.test.junit4)
    debugImplementation(libs.androidx.ui.tooling)
    debugImplementation(libs.androidx.ui.test.manifest)

    // In case we want to use VOSK
    implementation(libs.vosk.android)

    // For networking
    implementation(libs.retrofit)
    implementation(libs.retrofit.converter.moshi)
    implementation(libs.okhttp)
    implementation(libs.okhttp.logging.interceptor)

    // For JSON parsing with Moshi
    implementation(libs.json)
    implementation(libs.moshi)
    ksp(libs.moshi.compiler)

    // For Kotlin Coroutines
    implementation(libs.coroutines.android)
    implementation(libs.coroutines.core)
}