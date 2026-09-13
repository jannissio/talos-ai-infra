'use strict';
const recordings={
  demo:{src:'assets/Talos-demo-v6.mp4',caption:'Six learned dinner skills and table-supported bottle relays from recorded physics. Accelerated segments are labeled. The 3:09 presentation uses explanatory captions and has no audio track.'},
  dinner:{src:'assets/Talos-dinner-ten-v6.mp4',caption:'All ten frozen six-skill dinner sequences, at labeled 4× playback. Eight pass; one mug and one spoon placement fail the unchanged 8 mm threshold. These are finite task-start variations.'},
  relays:{src:'assets/Talos-relays-ten-v3.mp4',caption:'All ten frozen learned table-supported relay trials at 1× playback: five of five pass in each direction. Release onto the table separates the two arms’ grasps; no airborne exchange is claimed.'},
  composed:{src:'assets/Talos-composed-dinner-ten-v1.mp4',caption:'All ten frozen camera-planned workflows from the left-reach bottle region, at labeled 4× playback. Eight pass. All ten complete both relay legs and every task through fork; two spoon placements fail.'},
  rgb:{src:'assets/rgb-feedback-experiment.mp4',caption:'Unpromoted RGB experiment, V1 only: one pair selected after evaluation and shown at 1× playback, with the complete 48-trial counts. Live images pass 2/12 nominal and 3/12 pushed scenes. This is limited correction evidence, not broad bottle coverage.'}
};
const video=document.querySelector('#demo-video');
document.querySelectorAll('[data-video]').forEach(button=>button.addEventListener('click',()=>{
  const recording=recordings[button.dataset.video];
  video.pause();video.src=recording.src;video.removeAttribute('poster');video.load();
  document.querySelector('#video-caption').textContent=recording.caption;
  document.querySelector('#download-video').href=recording.src;
  document.querySelectorAll('[data-video]').forEach(other=>{
    const selected=other===button;other.classList.toggle('selected',selected);other.setAttribute('aria-pressed',String(selected));
  });
}));
