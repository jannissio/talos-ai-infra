'use strict';
const recordings={
  demo:{src:'assets/Talos-demo.mp4',caption:'Published baseline recording. Programmed table setting and relay are distinguished from learned bottle control; accelerated segments are labeled.'},
  seeds:{src:'assets/Talos-ten-seeds.mp4',caption:'All ten declared upright-bottle trials from the preserved model, shown together at original simulation speed. These cover small starting-position changes, not the full reachable workspace.'}
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
