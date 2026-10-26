// ---------- 360 뷰어 (WebGL, 라이브러리 없음)
export function createViewer(cv) {
  let gl, prog, tex, yaw = 0, pitch = 0, fov = 90, drag = null, ready = false;
  let loadVersion = 0;
  const uniforms = {};
  const vs = `attribute vec2 p;varying vec2 v;void main(){v=p;gl_Position=vec4(p,0.,1.);}`;
  const fs = `precision highp float;varying vec2 v;uniform sampler2D t;uniform float yaw,pitch,th,asp;
    void main(){vec3 d=normalize(vec3(v.x*th*asp,v.y*th,1.));
      float cp=cos(pitch),sp=sin(pitch);d=vec3(d.x,d.y*cp+d.z*sp,-d.y*sp+d.z*cp);
      float cy=cos(yaw),sy=sin(yaw);d=vec3(d.x*cy+d.z*sy,d.y,-d.x*sy+d.z*cy);
      float lon=atan(d.x,d.z),lat=asin(clamp(d.y,-1.,1.));
      gl_FragColor=texture2D(t,vec2(fract(lon/6.2831853+.5),.5-lat/3.1415927));}`;
  function setup() {
    gl = cv.getContext("webgl"); if (!gl) return false;
    const sh = (type, src) => { const s = gl.createShader(type); gl.shaderSource(s, src); gl.compileShader(s); return s; };
    prog = gl.createProgram(); gl.attachShader(prog, sh(gl.VERTEX_SHADER, vs)); gl.attachShader(prog, sh(gl.FRAGMENT_SHADER, fs));
    gl.linkProgram(prog); gl.useProgram(prog);
    for (const name of ["yaw", "pitch", "th", "asp"]) uniforms[name] = gl.getUniformLocation(prog, name);
    const b = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, b);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1,-1,1,-1,-1,1,1,1]), gl.STATIC_DRAW);
    const loc = gl.getAttribLocation(prog, "p"); gl.enableVertexAttribArray(loc); gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);
    tex = gl.createTexture(); gl.bindTexture(gl.TEXTURE_2D, tex);
    [gl.TEXTURE_WRAP_S, gl.TEXTURE_WRAP_T].forEach(k => gl.texParameteri(gl.TEXTURE_2D, k, gl.CLAMP_TO_EDGE));
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    cv.addEventListener("contextmenu", e => e.preventDefault());   // 미리보기 화면 저장 방지 (작은 해상도라 혼동)
    cv.addEventListener("pointerdown", e => { drag = [e.clientX, e.clientY, yaw, pitch]; cv.setPointerCapture(e.pointerId); });
    cv.addEventListener("pointermove", e => { if (!drag) return; const k = fov / cv.clientHeight * Math.PI / 180;
      yaw = drag[2] - (e.clientX - drag[0]) * k; pitch = Math.max(-1.5, Math.min(1.5, drag[3] + (e.clientY - drag[1]) * k)); draw(); });
    cv.addEventListener("pointerup", () => drag = null);
    cv.addEventListener("pointercancel", () => drag = null);
    cv.addEventListener("wheel", e => { e.preventDefault(); fov = Math.max(30, Math.min(120, fov + e.deltaY * 0.05)); draw(); }, {passive: false});
    window.addEventListener("resize", draw);
    return true;
  }
  function draw() {
    if (!ready) return;
    const w = Math.round(cv.clientWidth * devicePixelRatio), h = Math.round(cv.clientHeight * devicePixelRatio);
    if (!w || !h) return;
    if (cv.width !== w || cv.height !== h) { cv.width = w; cv.height = h; }
    gl.viewport(0, 0, w, h);
    gl.uniform1f(uniforms.yaw, yaw); gl.uniform1f(uniforms.pitch, pitch);
    gl.uniform1f(uniforms.th, Math.tan(fov * Math.PI / 360)); gl.uniform1f(uniforms.asp, w / h);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  }
  return {
    hide() {
      loadVersion++;
      ready = false;
      drag = null;
      cv.style.display = "none";
    },
    load(url) {
    const version = ++loadVersion;
    ready = false;
    cv.style.display = "block";
    if (!gl && !setup()) { cv.style.display = "none"; return; }
    const im = new Image(); im.onload = () => {
      if (version !== loadVersion) return;
      let src = im; const max = gl.getParameter(gl.MAX_TEXTURE_SIZE);
      if (Math.max(im.width, im.height) > max) {
        const c = document.createElement("canvas"), scale = max / Math.max(im.width, im.height);
        c.width = Math.max(1, Math.round(im.width * scale)); c.height = Math.max(1, Math.round(im.height * scale));
        c.getContext("2d").drawImage(im, 0, 0, c.width, c.height); src = c;
      }
      gl.bindTexture(gl.TEXTURE_2D, tex); gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGB, gl.RGB, gl.UNSIGNED_BYTE, src);
      ready = true; draw(); };
    im.onerror = () => { if (version === loadVersion) cv.style.display = "none"; };
    im.src = url;
  }};
}
