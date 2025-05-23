package com.example.demospeechtoform

import com.squareup.moshi.Json
import com.squareup.moshi.JsonClass

/**
 * Data class representing one item in the "requested_informations" list.
 * This is used both in the schema loaded from assets and in the API request.
 * It includes fields like title, description, type, and potentially nested groups or lists.
 */
@JsonClass(generateAdapter = true)
data class RequestedInformationItem(
    @Json(name = "title") val title: String,
    @Json(name = "description") val description: String,
    @Json(name = "type") val type: String,
    @Json(name = "accepted_values") val acceptedValues: List<String>? = null,
    @Json(name = "example") val example: String? = null,
    @Json(name = "group") val group: List<RequestedInformationItem>? = null,
    @Json(name = "list") val list: ListItemDefinition? = null
)

/**
 * Defines the structure for items within a "list" in RequestedInformationItem.
 */
@JsonClass(generateAdapter = true)
data class ListItemDefinition(
    @Json(name = "title") val title: String,
    @Json(name = "description") val description: String,
    @Json(name = "group") val group: List<RequestedInformationItem> // The items within the list are defined by a group
)

/**
 * Data class for parsing the entire report_schema.json file from assets.
 */
@JsonClass(generateAdapter = true)
data class ReportSchemaAsset(
    @Json(name = "requested_informations") val requestedInformations: List<RequestedInformationItem>,
    @Json(name = "topic") val topic: String,
    @Json(name = "language") val language: String
)

/**
 * Data class for the actual request payload to be sent to the API.
 * It includes the transcribed document and the schema information.
 */
@JsonClass(generateAdapter = true)
data class ApiRequest(
    @Json(name = "document") val document: String,
    @Json(name = "requested_informations") val requestedInformations: List<RequestedInformationItem>,
    @Json(name = "topic") val topic: String,
    @Json(name = "language") val language: String
)

// The API success response is a dynamic map (Map<String, Any?>),
// so we don't define a strict data class for the success response body itself.
// However, we can define a class for potential error responses.

/**
 * Data class for representing a potential error response from the API.
 * This is a generic example.
 */
@JsonClass(generateAdapter = true)
data class ApiError(
    @Json(name = "error") val error: String?, // Example field
    @Json(name = "message") val message: String?, // Example field
    @Json(name = "status_code") val statusCode: Int?, // Example field
    // Add other fields if the API returns more structured errors
    // e.g., @Json(name = "details") val details: Any?
)
