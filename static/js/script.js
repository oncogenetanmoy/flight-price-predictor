document.addEventListener('DOMContentLoaded', function() {
    // Get all images with the class 'homepage-image'
    const images = document.querySelectorAll('.homepage-image');
    let currentIndex = 0; // Start with the first image

    // Function to show the next image
    function showNextImage() {
        // Remove 'active' class from the current image
        images[currentIndex].classList.remove('active');

        // Calculate the index of the next image
        currentIndex = (currentIndex + 1) % images.length;

        // Add 'active' class to the next image
        images[currentIndex].classList.add('active');
    }

    // Start the slideshow - change image every 5000 milliseconds (5 seconds)
    setInterval(showNextImage, 3000); // Adjust the interval time (in milliseconds) as needed
});