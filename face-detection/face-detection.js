const video = document.getElementById("video");

function startVideo() {
  const mediaSource = new MediaSource();
  navigator.getUserMedia(
    { video: true },
    stream => (video.srcObject = mediaSource),
    err => console.error(err)
  );
}

startVideo();
