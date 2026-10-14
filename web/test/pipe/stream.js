// StreamSession replay over JS recognition results.  jsc <src> ser.js stream.js -- <fixtures dir> <set> <mode>
//   mode "video":  one session over all frames (manifest times), commits + revisions + finish   (what replay_video does)
//   mode "devlog": a fresh session per frame fed 6 copies of the same result, then finish        (a fused single-frame sheet)
(function (A) {
    var dir = A[0], set = A[1], mode = A[2], T = OMR.types;
    var man = JSON.parse(readFile(dir + '/' + set + '.manifest.json')), js = JSON.parse(readFile(dir + '/' + set + '.js.json'));
    function rec(kind, t, r, extra) {
        var o = { kind: kind, t: t, student_id: r.student_id, subject: r.subject, serial: r.serial, page: r.page, form_key: r.form_key,
            fused_frames: r.fused_frames, flags: r.flags.slice(), needs_review: T.needsReview(r),
            states: r.answers.map(function (a) { return [a.number, a.state, a.marked.slice(), !!a.ambiguous, a.guess]; }),
            id_values: r.id_digits.map(function (d) { return d.value; }), subject_values: r.subject_digits.map(function (d) { return d.value; }) };
        for (var k in (extra || {})) o[k] = extra[k];
        return o;
    }
    var out = [];
    if (mode === 'video') {
        var s = new OMR.stream.StreamSession({}), last = 0;
        for (var i = 0; i < man.length; i++) {
            var r = PIPE.rebuild(js[man[i].name]), t = man[i].t;
            var ev = s.feed(r, t, null);
            last = t;
            if (ev.commit) out.push(rec('commit', t, ev.commit.result, { frames: ev.commit.frames, frame_index: ev.commit.frame_index }));
            else if (ev.revision) out.push(rec('revision', t, ev.revision.result, { frames: ev.revision.frames, changed: ev.revision.changed }));
        }
        var c = s.finish(last);
        if (c) out.push(rec('finish', last, c.result, { frames: c.frames }));
        out.push({ kind: 'missed', episodes: s.unreadEpisodes(true).length, missed: s.missed(true).length });
    } else {
        for (var j = 0; j < man.length; j++) {
            var s2 = new OMR.stream.StreamSession({}), got = null;
            for (var q = 0; q < 6; q++) {
                var ev2 = s2.feed(PIPE.rebuild(js[man[j].name]), q / 8, null);
                if (ev2.commit && !got) got = rec('commit', q / 8, ev2.commit.result, { name: man[j].name });
            }
            if (!got) { var c2 = s2.finish(1); if (c2) got = rec('finish', 1, c2.result, { name: man[j].name }); }
            out.push(got || { kind: 'none', name: man[j].name });
        }
    }
    print('@' + JSON.stringify(out));
})(arguments);
