document.addEventListener('DOMContentLoaded', function() {
    // Define the list of image sources
    const imageSources = [
        'static/images/image1.jpg',
        'static/images/image2.jpg',
        'static/images/image3.jpg',
        'static/images/image4.jpg',
        'static/images/image5.jpg',
        'static/images/image6.jpg',
        'static/images/image7.jpg'

        // Add more image sources as needed
    ];

    // Function to rotate main images automatically
    function rotateImages() {
        const mainImage = document.querySelector('#gallery .main-image');
        let currentIndex = imageSources.indexOf(mainImage.getAttribute('src'));
        let nextIndex = (currentIndex + 1) % imageSources.length;
        mainImage.setAttribute('src', imageSources[nextIndex]);
    }

    // Rotate images automatically every 2 seconds
    setInterval(rotateImages, 2000);
});

document.getElementById('imageLink').onclick = function(event) {
  event.preventDefault(); // Prevent the default link behavior

  const imageWindow = window.open(this.href, 'Image', 'width=500,height=500'); // Open the image in a new popup window

  setTimeout(function() {
    imageWindow.close(); // Close the popup after 2 seconds
  }, 2000); // 2000 milliseconds = 2 seconds
};