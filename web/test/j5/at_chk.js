var man = JSON.parse(readFile('web/test/fixtures/pipe/video.manifest.json'))[16];
var gray = { w: man.w, h: man.h, data: readFile('web/test/fixtures/pipe/' + man.file, 'binary') };
function t(name, f, n) { var b = 1e9; for (var i = 0; i < (n||15); i++) { var t0 = performance.now(); f(); b = Math.min(b, performance.now()-t0);} print(name, b.toFixed(1)); }
var seed=7; function rnd(){ seed=(seed*1103515245+12345)&0x7fffffff; return seed; }
var bad=0;
[[30,20],[25,40],[100,3],[64,64],[27,5],[26,30]].forEach(function(wh){ [3,5,13,23,25].forEach(function(k){ var w=wh[0],h=wh[1]; if (w<k+2) return; var d=new Uint8Array(w*h); for (var i=0;i<d.length;i++) d[i]=rnd()%256; var img={w:w,h:h,data:d};
  var a=new Int8Array((w+2)*(h+2)), b=new Int8Array((w+2)*(h+2)); OMR.cv.adaptiveThreshold(img,k,7,a,true);
  // reference: unpadded plain
  var ref=OMR.cv.adaptiveThreshold(img,k,7,null,false);
  for (var y=0;y<h;y++) for (var x=0;x<w;x++) if ((ref.data[y*w+x]?1:0)!==a[(y+1)*(w+2)+x+1]) { bad++; }
  }); });
print('small cases bad', bad);
var buf = new Int8Array((gray.w+2)*(gray.h+2));
OMR.cv.adaptiveThreshold(gray,13,7,buf,true); var ref=OMR.cv.adaptiveThreshold(gray,13,7,null,false), b2=0; for (var y=0;y<gray.h;y++) for (var x=0;x<gray.w;x++) if ((ref.data[y*gray.w+x]?1:0)!==buf[(y+1)*(gray.w+2)+x+1]) b2++; print('frame bad', b2);
t('fused 13', function(){ OMR.cv.adaptiveThreshold(gray,13,7,buf,true); });
t('plain 13', function(){ OMR.cv.adaptiveThreshold(gray,13,7,null,false); });
