// jsc web/test/app/syntax.js -- file1.js file2.js ... : parse each file (no execution) with new Function
var bad = 0;
for (var i = 0; i < arguments.length; i++) {
    try { new Function(readFile(arguments[i])); print('ok   ' + arguments[i]); }
    catch (e) { bad++; print('FAIL ' + arguments[i] + ': ' + e); }
}
if (bad) throw new Error(bad + ' syntax failures');
