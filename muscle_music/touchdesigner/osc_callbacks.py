"""Callbacks DAT for a TouchDesigner OSC In DAT listening on UDP 9001."""


LAYER_NAMES = ("drums", "texture", "chords", "lead", "arp", "bass")


def _state_table():
    table = op("jilv_state")
    if table is None:
        return None
    if table.numRows == 0:
        table.appendRow(["key", "value"])
    return table


def _put(key, value):
    table = _state_table()
    if table is None:
        return
    cell = table.findCell(str(key), cols=[0])
    if cell is None:
        table.appendRow([str(key), str(value)])
    else:
        table[cell.row, 1] = str(value)


def onReceiveOSC(dat, rowIndex, message, bytes, timeStamp, address, args, peer):
    del dat, rowIndex, message, bytes, timeStamp, peer
    if address == "/music/gesture":
        _put("gesture_id", int(args[0]))
        _put("gesture_confidence", float(args[1]))
        _put("motion_energy", float(args[2]))
        _put("emg_activation", float(args[3]))
    elif address == "/music/transport":
        _put("transport", args[0])
        _put("countdown", int(args[1]))
        _put("bpm", float(args[2]))
    elif address == "/music/beat":
        _put("bar", int(args[0]))
        _put("beat", int(args[1]))
        _put("subdivision", int(args[2]))
        _put("beat_pulse", absTime.frame)
    elif address == "/music/layer":
        layer = int(args[0])
        prefix = "layer_{}".format(layer)
        _put(prefix + "_name", LAYER_NAMES[layer])
        _put(prefix + "_active", int(args[1]))
        _put(prefix + "_generation", int(args[2]))
        _put(prefix + "_stage", int(args[3]))
        _put(prefix + "_energy", float(args[4]))
        _put(prefix + "_change", args[5])
        _put(prefix + "_pulse", absTime.frame)
    elif address == "/music/harmony":
        _put("root", args[0])
        _put("mode", args[1])
    elif address == "/music/personality":
        _put("personality_label", args[0])
        _put("personality_motion", float(args[1]))
        _put("personality_muscle", float(args[2]))
        _put("personality_legato", float(args[3]))
        _put("personality_complexity", float(args[4]))
    elif address == "/music/variation":
        _put("variation", args[0])
        _put("variation_pulse", absTime.frame)
    elif address == "/music/reset":
        _put("reset_pulse", absTime.frame)
    elif address == "/music/error":
        _put("error", args[0])
    return


def onReceiveError(dat, rowIndex, message, bytes, timeStamp, address, args, peer):
    del dat, rowIndex, bytes, timeStamp, address, args, peer
    _put("error", message)
    return
