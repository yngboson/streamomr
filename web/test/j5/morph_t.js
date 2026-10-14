// brute-force check of erode/dilate/morphOpen U8 against the definition
function brute(img, k, isMax) { var w=img.w,h=img.h,a=k>>1,o=new Uint8Array(w*h);
  for (var y=0;y<h;y++) for (var x=0;x<w;x++){ var m=isMax?0:255; for (var j=y-a;j<=y+k-1-a;j++) for (var i=x-a;i<=x+k-1-a;i++){ if(i<0||j<0||i>=w||j>=h) continue; var v=img.data[j*w+i]; if(isMax? v>m : v<m) m=v; } o[y*w+x]=m; } return o; }
var seed=12345; function rnd(){ seed=(seed*1103515245+12345)&0x7fffffff; return seed; }
var bad=0, cnt=0;
[[7,5],[8,9],[13,11],[31,17],[33,40],[1,1],[2,6],[50,3],[64,21]].forEach(function(wh){ [1,2,3,4,5,8,9,13,15,16,17,24].forEach(function(k){
  var w=wh[0],h=wh[1], d=new Uint8Array(w*h); for (var i=0;i<d.length;i++) d[i]=rnd()%256; var img={w:w,h:h,data:d};
  [true,false].forEach(function(mx){ var r=(mx?OMR.cv.dilate:OMR.cv.erode)(img,k), b=brute(img,k,mx); cnt++; for (var i=0;i<b.length;i++) if (r.data[i]!==b[i]) { bad++; print('BAD',w,h,k,mx); break; } });
  var op=OMR.cv.morphOpen(img,k), e=brute(img,k,false), e2={w:w,h:h,data:e}, ob=brute(e2,k,true); cnt++; for (var i=0;i<ob.length;i++) if(op.data[i]!==ob[i]){bad++;print('BADOPEN',w,h,k);break;}
}); });
print('checked', cnt, 'bad', bad);
