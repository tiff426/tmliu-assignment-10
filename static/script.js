// document.getElementById("query-form").addEventListener("submit", async function(event) {
//     event.preventDefault(); // Prevent default form submission

//     // Get elements
//     const queryType = document.getElementById("query-type").value;
//     const textQuery = document.getElementById("text-query").value.trim();
//     const imageQuery = document.getElementById("image-query").files[0];
//     const weight = parseFloat(document.getElementById("weight").value);
//     const embeddingType = document.getElementById("embedding-type").value;
//     const resultsSection = document.getElementById("results-section");
//     const resultsList = document.getElementById("results-list");

//     // Reset previous results
//     resultsSection.style.display = "none";
//     resultsList.innerHTML = "";

//     // Validation based on query type
//     if (queryType === "text" && !textQuery) {
//         alert("Please enter a text query.");
//         return;
//     }

//     if (queryType === "image" && !imageQuery) {
//         alert("Please upload an image.");
//         return;
//     }

//     if (queryType === "hybrid") {
//         if (!textQuery || !imageQuery) {
//             alert("Please provide both text query and an image for a hybrid query.");
//             return;
//         }
//         if (isNaN(weight) || weight < 0.0 || weight > 1.0) {
//             alert("Please enter a weight between 0.0 and 1.0 for the hybrid query.");
//             return;
//         }
//     }

//      // Prepare form data for the request
//     const formData = new FormData();
//     formData.append("query_type", queryType);
//     formData.append("text_query", textQuery);
//     if (imageQuery) {
//         formData.append("image_query", imageQuery);
//     }
//     if (queryType === "hybrid") {
//         formData.append("weight", weight);
//     }
//     formData.append("embedding_type", embeddingType);

//     // Send the request to the backend
//     try {
//         const response = await fetch("/image_search", {
//             method: "POST",
//             body: formData,
//         });

//         if (!response.ok) {
//             throw new Error("Failed to fetch results.");
//         }

//         const data = await response.json();

//         // Display results
//         if (data.results && data.results.length > 0) {
//             resultsSection.style.display = "block";
//             data.results.forEach((result) => {
//                 const listItem = document.createElement("li");
//                 listItem.innerHTML = `
//                     <strong>Image:</strong> 
//                     <img src="${result.image_url}" alt="Result Image" style="max-width: 100px;"> 
//                     <br><strong>Score:</strong> ${result.similarity_score}`;
//                 resultsList.appendChild(listItem);
//             });
//         } else {
//             resultsSection.style.display = "block";
//             resultsList.innerHTML = "<li>No results found.</li>";
//         }
//     } catch (error) {
//         console.error("Error fetching results:", error);
//         alert("An error occurred while fetching results.");
//     }
// });
document.getElementById('query-form').addEventListener('submit', function (event) {
    event.preventDefault();

    let queryType = document.getElementById('query-type').value;
    let textQuery = document.getElementById('text-query').value.trim();
    let imageQuery = document.getElementById('image-query').files[0];
    let weight = parseFloat(document.getElementById('weight').value);
    let embeddingType = document.getElementById('embedding-type').value;
    let resultsDiv = document.getElementById('results');
    resultsDiv.innerHTML = '';

    // Validation based on query type
    if (queryType === "text" && !textQuery) {
        alert("Please enter a text query.");
        return;
    }

    if (queryType === "image" && !imageQuery) {
        alert("Please upload an image.");
        return;
    }

    if (queryType === "hybrid") {
        if (!textQuery || !imageQuery) {
            alert("Please provide both text query and an image for a hybrid query.");
            return;
        }
        if (isNaN(weight) || weight < 0.0 || weight > 1.0) {
            alert("Please enter a weight between 0.0 and 1.0 for the hybrid query.");
            return;
        }
    }

    // Prepare form data for the request
    const formData = new FormData();
    formData.append("query_type", queryType);
    formData.append("text_query", textQuery);
    if (imageQuery) {
        formData.append("image_query", imageQuery);
    }
    if (queryType === "hybrid") {
        formData.append("weight", weight);
    }
    formData.append("embedding_type", embeddingType);

    // Send the request to the backend
    fetch('/image_search', {
        method: 'POST',
        body: formData,
    })
    .then(response => response.json())
    .then(data => {
        console.log(data);
        displayResults(data);
    });
});

function displayResults(data) {
    let resultsDiv = document.getElementById('results');
    resultsDiv.innerHTML = '<h2>Results</h2>';
    if (data.images && data.images.length > 0) {
        for (let i = 0; i < data.images.length; i++) {
            let docDiv = document.createElement('div');
            docDiv.innerHTML = `<strong>Image <img src="./coco_images_resized/${data.images[i]}" alt="OH MY GODDDRWFFECWE" style="max-width: 100px;"></strong><br><strong>Similarity: ${data.top_sims[i]}</strong>`;
            resultsDiv.appendChild(docDiv);
        }
    } else {
        resultsDiv.innerHTML = '<p>No results found.</p>';
    }
}
