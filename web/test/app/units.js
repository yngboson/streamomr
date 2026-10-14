// jsc <src files...> web/test/app/units.js -- <rgba.bin> <w> <h> <grey.bin> <gw> <gh>
// prints @{"gray": hex of rgbaToGray, "jpeg": [base64 of encodeJpegGray at q75 and q90]}
(function (A) {
    var rgba = readFile(A[0], 'binary'), w = +A[1], h = +A[2], g = readFile(A[3], 'binary'), gw = +A[4], gh = +A[5];
    var gray = OMR.scanner.rgbaToGray(rgba, w, h).data, hex = [];
    for (var i = 0; i < gray.length; i++) hex.push((gray[i] < 16 ? '0' : '') + gray[i].toString(16));
    var img = { w: gw, h: gh, data: g };
    print('@' + JSON.stringify({ gray: hex.join(''), jpeg: [OMR.scanner.b64(OMR.scanner.encodeJpegGray(img, 75)), OMR.scanner.b64(OMR.scanner.encodeJpegGray(img, 90))] }));
})(arguments);
