const total_number_images = 1000;

function update_image(redirect) {
    var storageRef = firebase.storage().ref();
    var imagesRef = storageRef.child('images');
    imagesRef.child(get_random_image_id()).getDownloadURL().then(function(url) {
        if (redirect) {
            window.location.href = url;
        } else {
            document.querySelector('#imgDisplay').setAttribute('src', url);
        }
    }).catch(function(error) {});
}

function get_random_image_id() {
    var img_id = (Math.floor(Math.random() * total_number_images) + 1).toString();
    img_id = "seed".concat("0".repeat(4-img_id.length).concat(img_id.concat('.jpg')));
    return img_id
}