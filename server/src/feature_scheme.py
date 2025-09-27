import pefile
import numpy as np


class PE:
    MAX_SECTION_BYTES = 2 * 1024 * 1024   # cap entropy analysis at 2 MB
    MAX_TLS_CALLBACKS = 1000
    MAX_RESOURCE_DEPTH = 3
    MAX_RESOURCE_ENTRIES = 5000

    def __init__(self, path):
        self.path = path
        try:
            self.pe = pefile.PE(path, fast_load=False)

            if self.pe.FILE_HEADER.Characteristics & 0x2000:  # IMAGE_FILE_DLL
                raise ValueError("File is a DLL, not an EXE")
        except Exception as e:
            raise ValueError(f"Failed to parse PE: {e}")

    # ------------------- Safe getters -------------------

    def get_size_of_code(self):
        return getattr(self.pe.OPTIONAL_HEADER, "SizeOfCode", 0)

    def get_number_of_sections(self):
        return getattr(self.pe.FILE_HEADER, "NumberOfSections", 0)

    def get_timedatestamp(self):
        return getattr(self.pe.FILE_HEADER, "TimeDateStamp", 0)

    def get_number_of_symbols(self):
        return getattr(self.pe.FILE_HEADER, "NumberOfSymbols", 0)

    def get_machine(self):
        return getattr(self.pe.FILE_HEADER, "Machine", 0)

    def get_size_of_initialized_data(self):
        return getattr(self.pe.OPTIONAL_HEADER, "SizeOfInitializedData", 0)

    def get_size_of_uninitialized_data(self):
        return getattr(self.pe.OPTIONAL_HEADER, "SizeOfUninitializedData", 0)

    def get_size_of_image(self):
        return getattr(self.pe.OPTIONAL_HEADER, "SizeOfImage", 0)

    def get_size_of_stack_reserve(self):
        return getattr(self.pe.OPTIONAL_HEADER, "SizeOfStackReserve", 0)

    def get_checksum(self):
        return getattr(self.pe.OPTIONAL_HEADER, "CheckSum", 0)

    def get_address_of_entry_point(self):
        return getattr(self.pe.OPTIONAL_HEADER, "AddressOfEntryPoint", 0)

    def get_subsystem(self):
        return getattr(self.pe.OPTIONAL_HEADER, "Subsystem", 0)

    # ------------------- Section info -------------------

    def _safe_entropy(self, data: bytes):
        if not data:
            return 0.0
        # cap size
        if len(data) > self.MAX_SECTION_BYTES:
            data = data[:self.MAX_SECTION_BYTES]
        freq = np.bincount(np.frombuffer(data, dtype=np.uint8), minlength=256)
        probs = freq / len(data)
        return float(-np.sum([p * np.log2(p) for p in probs if p > 0]))

    def get_section_info(self, section_name):
        info = {
            f"{section_name}_exists": False,
            f"{section_name}_SizeOfRawData": 0,
            f"{section_name}_entropy": 0,
        }
        for section in getattr(self.pe, "sections", []):
            name = section.Name.decode("utf-8", errors="ignore").strip("\x00")
            if name == section_name:
                info[f"{section_name}_exists"] = True
                info[f"{section_name}_SizeOfRawData"] = section.SizeOfRawData
                entropy = self._safe_entropy(section.get_data())
                info[f"{section_name}_entropy"] = entropy
                break
        return info
    
    def get_major_linker_version(self): 
        linker_version_scores = {0: -0.23947141660483862, 1: -0.38697351739168695, 2: -0.5134038894946998, 3: 0.055532784968858086, 4: -0.1481605923082182, 5: -0.39399742695296547, 6: 1.235549591263645, 7: -2.5995050291944124, 8: -0.3518539695852945, 9: 0.7087563741677579, 10: 3.855467857620523, 11: -0.4642365225657504, 12: -1.0612688352744224, 13: -0.22542359748228163, 14: 1.523529883276063, 15: -0.19030404967588918, 16: -0.1832801401146107, 17: -0.2535192357273956, 18: -0.2535192357273956, 19: -0.1832801401146107, 27: -0.1551845018694967, 33: -0.20435186879844616, 34: -0.23244750704356013, 48: 1.7974623561659244, 54: -0.20435186879844616, 71: -0.20435186879844616, 76: -0.20435186879844616, 80: 0.027437146723744112, 83: -0.20435186879844616, 89: -0.2535192357273956, 133: -0.19030404967588918, 255: 0.09767624233652904}
        version = self.pe.OPTIONAL_HEADER.MajorLinkerVersion
        return linker_version_scores.get(version, 0)

    def get_minor_linker_version(self):
        linker_version_scores = {25: 4.317887966945434, 20: -0.5034200224814006, 0: 5.251044351995788, 39: -0.5889593577776832, 50: -2.424166915043381, 43: -0.15348637808751742, 13: 1.2151429866530032, 10: -1.1177479759728843, 32: 0.9196434647203908, 11: -0.7678143315790011, 40: -0.4723148096463887, 29: -0.31678874547132957, 12: -0.3012361390538237, 27: -0.9622219117978251, 42: -0.7444854219527423, 38: -0.5500778417339184, 22: 0.033144898922553595, 35: -0.06794704279123487, 16: -0.16126268129627036, 31: -0.1379337716700115, 3: 0.01759229250504768, 34: 0.9896301935991675, 28: 0.04869750534005952, 56: -0.26235462301005885, 44: -0.08349964920874078, 71: -0.3245650486800825, 30: 0.34419702727267193, 9: -0.04461813316497599, 36: -0.052394436373728946, 37: -0.3478939583063414, 7: 0.29753920802015416, 15: -0.10682855883499966, 51: -0.10682855883499966, 17: -0.06017073958248191, 41: -0.15348637808751742, 1: -0.013512920329964156, 24: -0.14571007487876445, 55: -0.16126268129627036, 4: 0.06425011175756543, 33: -0.04461813316497599, 160: -0.09127595241749374, 58: -0.1379337716700115, 8: -0.10682855883499966, 14: -0.10682855883499966, 21: -0.15348637808751742, 66: -0.12238116525250557, 6: -0.021289223538717116, 65: -0.12238116525250557, 5: -0.04461813316497599, 2: -0.10682855883499966, 26: -0.12238116525250557, 82: -0.12238116525250557, 88: -0.10682855883499966, 111: -0.12238116525250557, 57: -0.13015746846125853, 86: -0.12238116525250557, 165: -0.12238116525250557, 95: -0.10682855883499966, 122: -0.12238116525250557, 47: -0.12238116525250557, 89: -0.10682855883499966}
        version = self.pe.OPTIONAL_HEADER.MinorLinkerVersion
        return linker_version_scores.get(version, 0)

    def get_section_anomalies(self):
        anomalies = {
            "section_executable": 0,
            "section_writable": 0,
            "section_rwx": 0,
            "sections_entropy_high": 0,
        }
        for section in getattr(self.pe, "sections", []):
            c = section.Characteristics
            is_exec = c & 0x20000000 != 0
            is_write = c & 0x80000000 != 0
            is_read = c & 0x40000000 != 0
            if is_exec:
                anomalies["section_executable"] += 1
            if is_write:
                anomalies["section_writable"] += 1
            if is_exec and is_write and is_read:
                anomalies["section_rwx"] += 1
            if self._safe_entropy(section.get_data()) >= 7.0:
                anomalies["sections_entropy_high"] += 1

        anomalies["section_executable"] = anomalies["section_executable"]
        anomalies["section_writable"] = anomalies["section_writable"]
        anomalies["section_rwx"] = anomalies["section_rwx"]
        anomalies["sections_entropy_high"] = anomalies["sections_entropy_high"]
        return anomalies

    # ------------------- Imports -------------------

    def get_imported_dlls_and_functions(self):
        dlls = funcs = 0
        try:
            for entry in getattr(self.pe, "DIRECTORY_ENTRY_IMPORT", []):
                dlls += 1
                funcs += len(entry.imports)
        except Exception:
            pass
        return {
            "ImportedDLLs": dlls,
            "ImportedFunctions": funcs,
        }

    def get_spyware_api_calls(self):
        suspicious = ['getmodulehandlea', 'getprocaddress', 'muldiv', 'readfile', 'setendoffile', 'regopenkeyexa', 'regqueryvalueexa', 'getmodulefilenamea', 'createfilea', 'writefile', 'sysfreestring', 'exitprocess', 'closehandle', 'setfilepointer', 'messageboxa', 'sleep', 'setcurrentdirectorya', 'createthread', 'freelibrary', 'charnexta', 'gettickcount', 'interlockeddecrement', 'regclosekey', 'multibytetowidechar', 'localalloc', 'getlocaleinfoa', 'virtualalloc', 'loadstringa', 'loadlibrarya', 'globalfindatoma', 'getthreadlocale', 'findfirstfilea', 'interlockedincrement', 'findclose', 'globaladdatoma', 'virtualquery', 'deletefilea', 'lstrcpyna', 'getversion', 'deletecriticalsection', 'regsetvalueexa', 'gettempfilenamea', 'isequalguid', 'entercriticalsection', 'leavecriticalsection', 'getcurrentdirectorya', 'regcreatekeyexa', 'getfilesize', 'setfiletime', 'openprocess', 'getcurrentthreadid', 'regflushkey', 'getlasterror', 'stringfromclsid', 'getstdhandle', 'charuppera', 'getusernamea', 'cotaskmemfree', 'gettemppatha', 'initializecriticalsection', 'couninitialize', 'getversionexa', 'setfileattributesa', 'tlssetvalue', 'getfileversioninfosizea', 'getfileversioninfoa', 'verqueryvaluea', 'findwindowa', 'exitthread', 'getdiskfreespacea', 'freeresource', 'waitforsingleobject', 'openprocesstoken', 'virtualfree', 'comparestringa', 'widechartomultibyte', 'findresourcea', 'winhelpa', 'virtualallocex', 'cocreateinstance', 'raiseexception', 'getmenuiteminfoa', 'findnextfilea', 'getwindowlonga', 'send', 'setwindowlonga', 'callwindowproca', 'getdateformata', 'postmessagea', 'socket', 'getclassnamea', 'formatmessagea', 'getuserdefaultlcid', 'msgwaitformultipleobjects', 'getbrushorgex', 'peekmessagea', 'insertmenuitema', 'getcomputernamea', 'destroycursor', 'variantclear', 'getobjecta', 'getstringtypeexa', 'closesocket', 'getdcorgex', 'setpropa', 'enumcalendarinfoa', 'createeventa', 'copyenhmetafilea', 'setwindowshookexa', 'getprocessheap', 'getcurrentprocessid', 'getfilesizeex', 'imagelist_setdragcursorimage', 'dispatchmessagea', 'imagelist_setbkcolor', 'setenhmetafilebits', 'playenhmetafile', 'getpropa', 'imagelist_getbkcolor', 'restoredc', 'insertmenua', 'defwindowproca', 'getenhmetafileheader', 'getclipboarddata', 'getcapture', 'registerclipboardformata', 'defframeproca', 'gettextmetricsa', 'createfontindirecta', 'activatekeyboardlayout', 'maskblt', 'getenvironmentvariablea', 'chartooema', 'createbitmap', 'setclasslonga', 'getmenustringa', 'imagelist_dragshownolock', 'gettextextentpoint32a', 'getbitmapbits', 'releasedc', 'getwindowtexta', 'imagelist_drawex', 'createmenu', 'globalrealloc', 'charlowera', 'getcurrentpositionex', 'bitblt', 'imagelist_dragmove', 'imagelist_dragenter', 'imagelist_enddrag', 'imagelist_begindrag', 'imagelist_dragleave', 'intersectcliprect', 'imagelist_getimagecount', 'getdcex', 'getmodulefilenamew', 'showcursor', 'getforegroundwindow', 'charlowerbuffa', 'createpenindirect', 'setstretchbltmode', 'lstrcpya', 'getenhmetafiledescriptiona', 'registerwindowmessagea', 'variantinit', 'globalhandle', 'getenhmetafilepaletteentries', 'getwinmetafilebits', 'createwindowexa', 'loadbitmapa', 'getenhmetafilebits', 'sysreallocstringlen', 'setwinmetafilebits', 'imagelist_getdragimage', 'wsagetlasterror', 'translatemdisysaccel', 'unrealizeobject', 'imagelist_write', 'getwindoworgex', 'gettopwindow', 'deletemenu', 'enablescrollbar', 'createicon', 'globalfree', 'globaldeleteatom', 'getsystempaletteentries', 'sendmessagea', 'safearraygetlbound', 'setthreadlocale', 'globalunlock', 'createdibitmap', 'imagelist_read', 'createhalftonepalette', 'getdibcolortable', 'removemenu', 'setviewportorgex', 'createenhmetafilea', 'safearrayptrofindex', 'getactivewindow', 'setwindoworgex', 'removepropa', 'variantchangetype', 'getfullpathnamea', 'getpaletteentries', 'imagelist_seticonsize', 'setrop2', 'setbrushorgex', 'isdialogmessagea', 'drawmenubar', 'defmdichildproca', 'rtlunwind', 'resumethread', 'createprocessa', 'savedc', 'showownedpopups', 'setmenuiteminfoa', 'getkeynametexta', 'selectpalette', 'setdibcolortable', 'deleteenhmetafile', 'setparent', 'drawicon', 'imagelist_remove', 'realizepalette', 'getscrollrange', 'getkeyboardlayoutlist', 'getclipbox', 'setwindowplacement', 'rectvisible', 'createbrushindirect', 'globalalloc', 'stretchblt', 'getkeyboardstate', 'registerclassa', 'imagelist_draw', 'regdeletevaluea', 'excludecliprect', 'loadcursora', 'getkeyboardlayout', 'drawtexta', 'getcursor', 'pathfileexistsa', 'globallock', 'safearraycreate', 'isaccelerator', 'createdibsection', 'getscrollinfo', 'oemtochara', 'loadkeyboardlayouta', 'copyfilea', 'setscrollpos', 'strstria', 'tlsgetvalue', 'inflaterect', 'destroyicon', 'progidfromclsid', 'shellexecuteexa', 'showscrollbar', 'gethostbyname', 'setactivewindow', 'isrectempty', 'getclassinfoa', 'getdrivetypea', 'resetevent', 'intersectrect', 'getwindowdc', 'getkeyboardtype', 'getdibits', 'safearraygetubound', 'olesetmenudescriptor', 'loadlibraryexa', 'wsastartup', 'regqueryinfokeya', 'imagelist_geticonsize', 'getexitcodethread', 'closeenhmetafile', 'safearraygetelement', 'getmenuitemid', 'createcompatiblebitmap', 'variantcopy', 'gettimezoneinformation', 'createpalette', 'setpixel', 'getpixel', 'drawiconex', 'lookupprivilegevaluea', 'waitmessage', 'mapvirtualkeya', 'getdesktopwindow', 'getlocaltime', 'process32first', 'equalrect', 'callnexthookex', 'geterrorinfo', 'lineto', 'unhookwindowshookex', 'setrect', 'createtoolhelp32snapshot', 'writeprocessmemory', 'getmenustate', 'setscrollinfo', 'movetoex', 'systemparametersinfoa', 'enumwindows', 'deletefilew', 'deleteobject', 'createpopupmenu', 'getcurrentdirectoryw', 'getscrollpos', 'windowfrompoint', 'lstrcmpa', 'unregisterclassa', 'imagelist_add', 'postquitmessage', 'adjustwindowrectex', 'movefilea', 'getcpinfo', 'setmenu', 'connect', 'getsystemmenu', 'localfree', 'getmenu', 'patblt', 'imagelist_destroy', 'enumthreadwindows', 'process32next', 'flushfilebuffers', 'dosdatetimetofiletime', 'isdebuggerpresent', 'setforegroundwindow', 'getmessagetime', 'inet_addr', 'createremotethread', 'lstrlena', 'scrollwindow', 'drawedge', 'getactiveobject', '_corexemain', 'setwindowtexta', 'drawframecontrol', 'globalsize', 'getvolumeinformationa', 'keybd_event', 'settimer', 'offsetrect', 'recv', 'writeprivateprofilestringa', 'setscrollrange', 'trackpopupmenu', 'senddlgitemmessagea', 'sysallocstringlen', 'wsaasyncgethostbyname', 'removedirectorya', 'ptinrect', 'seterrormode', 'createdirectorya', 'getsystemmetrics']
        count = 0
        try:
            for entry in getattr(self.pe, "DIRECTORY_ENTRY_IMPORT", []):
                for imp in entry.imports:
                    if imp.name:
                        fn = imp.name.decode("utf-8", errors="ignore").lower()
                        if fn in suspicious:
                            count += 1
        except Exception:
            pass
        return count

    # ------------------- Other features -------------------

    def get_overlay_size(self):
        if not getattr(self.pe, "sections", []):
            return 0
        last_end = max(s.PointerToRawData + s.SizeOfRawData for s in self.pe.sections)
        file_size = len(getattr(self.pe, "__data__", b""))
        return max(0, file_size - last_end)

    def get_score_suspicious_sections(self):        
        def levenshtein(s1, s2):
            if len(s1) < len(s2):
                return levenshtein(s2, s1)
            if len(s2) == 0:
                return len(s1)

            previous_row = list(range(len(s2) + 1))
            for i, c1 in enumerate(s1):
                current_row = [i + 1]
                for j, c2 in enumerate(s2):
                    insertions = previous_row[j + 1] + 1
                    deletions = current_row[j] + 1
                    substitutions = previous_row[j] + (c1 != c2)
                    current_row.append(min(insertions, deletions, substitutions))
                previous_row = current_row
            return previous_row[-1]
        
        def shannon_entropy(s):
            import math
            from collections import Counter

            if not s:
                return 0
            freq = Counter(s)
            probs = [count / len(s) for count in freq.values()]
            return -sum(p * math.log2(p) for p in probs)

        suspicious_names = [
            ".upx0", ".upx1", ".upx2", ".vmp0", ".vmp1", ".themida", ".mpress1", ".mpress2",
            "upx2", "upx0", "upx1", ".ropf", ".>fc", ".;{", ".n(", ".htext", "_rdata", ".vmp0", ".vmp1", ".ehkokk", ".wboyt", ".jvqpnf", ".magzg", ".imports", ".s", ".data1", ".nzx", ".px", ".tu", ".hdkod", ".uybwz", ".mpress2", ".eh_fram", ".code", "/102", ".ccc", ".mpress1", ".didat", ".mackt", "", "rt_code", ".00cfg", ".l", ".text3", ".dhgml", ".lzmjsu", ".yoswo", ".stdegl", ".fwqo", ".kubc", ".hkw", ".symtab", "fkp0", "fkp1", ".wzmlwv", ".vgz", ".bib", ".skvlv", ".x", ".adata", ".l1\u0000pi32", ".i", ".zeqopv", ".tixjl", ".m", ".qys", ".jvnc", ".bcq", ".gmfodl", ".ifpdc", ".ox", ".y", ".pnhk", ".ec", ".zuqj", ".kz", ".bzhvqq", ".fj", ".ookm", ".bdd", ".wggrl", ".dav", " ", "iokgdtty", ".textbss", ".fptable", ".bqwzja", ".wjtd", ".h", ".grrl", ".lx", ".eb", ".n", ".roszpwt", ".msvcjmc", ".d", ".debug_a", ".dijpp", ".gmd", ".pultca", ".yn", ".wkf", ".pd", ".mor", "hhjlvmvf", "obpgbsuh", "kekcblxo", "iakctuui", "bzezjopb", "dzcxfjgi", "ouhngryo", "wogipmel", "ziujaknn", "wbqcolky", ".themida", "jkghsgh", ".ma", ".obptk", ".wv", ".ueara", ".rddrej", ".roxfgq", ".bojdrg", ".iu", ".kzg", ".cmrd", ".bsp", ".j", ".sedata", ".zivvftt", ".dataf", "jhdfrwg", "d4", ".fen", ".cjwqrjx", ".taggant", ".ex", ".ah", ".wc", ".ckgaft", ".jbcccw", "ujhfftrt", "tgdserr", ".ugmfblr", "d2", ".nkytz", ".fc", ".w", ".voltbl", ".dbg", ".xjpy", ".didi", ".baheyop", ".yex", ".lire", ".wufopi", ".dbgfudv", ".sggsgds", ".drinkd", ".u'!<(", ".y)v$5", ".8", ".oy3", ".-!r", ".\"", ".5wvj%", ".debug_i", ".debug_l", ".debug_f", ".debug_s", ".hawke", ".vmp\ud83c\udf24", "dat", ".myreloc", ".x2\"*", ".d>ir", ".t:o", ".b9\"g9y", ".ljn", ".'", ".v7284", "!sugar", "$i^jte%h", ".khviwwg", "d3", ".dfqsvfp", ".vrqbaz", ".teext", ".otvwz", ".tbrkyi", ".hbfqm", "asdcvbgh", ".boot", ".xxxlgzy", ".12xja", "asddddd", ".zalfot", ".burns", ".xabk", ".ysti", ".ccigv", ".ebidd", ".1", ".xjneqsm", ".d11", ".zdat", ".kejlmks", ".t", ".gf", ".heyea", ".uh", ".tqqcdb", " \u0000 ", ".zochn1", ".p", ".\\jq", ".7|2", ".ruda", ".mybss", ".mydata", ".myedata", ".mycode", ".niko", ".bbs", ".vbbk", ".mytls", ".gurrvut", ".bxqnpqv", ".wgzpjsq", ".vmp@!@!", ".ozhbtym", "i/(\u0011:\u0001:,", ".rqns9vf", "vqrob", "vavqe", ".abr135", "natives~", "config~", "crypt~", ".idata ", ".lrzvny", ".lwbap", "ndehy", ".76b", ".@j'", "se", ".za|", ".'hs", ". wm", ".szs", ".hik", ".cuvujo", ".][0", "./+&", ".sxa", "rdyybf", ".eser", ".a %", ".^v7", ".>vy", ".t{u", ".qdata", "t", "\u0012\f", "{r$'", "nw\u001b~0", "9", "\u001b<", "ianixp", "ckikn", ".s2data", ".s1data", ".t6ata", ".t5ata", ".t4ata", ".t3ata", ".t2ata", ".t1ata", ".ahnisb", ".<jt", "xcuitb", ".suh", ".comulup", ".mos", "anlmql", ".ktays", ".febapeh", ".feva", ".mesorel", ".yafub", ".zqpgh", ".fomxo", ".u2b", ".c1(", ".q>", ".xj*", ".venue", "gfw", ".jap", ".d0h", ".>ha", ".bmf", ".vmp|^0", ".vmp|^1", ".vmp|^2", ".tfcrbd", ".huccq", ".idir", ".lakana", ".twmpq", ".aozxj", ".links", ".fq<", ".n/2", ".zykke", ".pesre", ".ig", ".li", ".lesepe", ".licegu", ".qv]", ".vjnlbf", ".kudd", ".fee", ".kbvkk", ".wkq", ".uxadns", ".jfpzob", ".cmf", ".<9;kn%", ".e", ".0<+'", ".<", ".fpfjb", ".(c1$7", "rx\u0010\u001ef\u001a<(", "rsmtc", ".<yx", ".fhi", ".gg2", ".3q_", "\u001cw'g", "0o", "\u0015a", "?k\u0001p]r", ".zoxav", ".chdma", ".kofwk", ".nvpmvbc", ".vgxl", ".kntpjtc", ".xnlx", ".qmydfcc", ".gdxpfcc", ".erxafcc", ".iopfela", ".jvhwf", ".3d<", ".qek", ".ejb", ".kopayow", ".foxer", ".podeje", "ygderrst", "arqygh", ".fuvitul", ".newiat", ".0ad", ".[k", ".ibp", ".&je", ".voqryb", ".wp", "d\u008b{l\u0001", "r\\\"a2", "epng", ".kah", ".@h*", ".t29", ".gfwe", ".tdata", ".5ue", "/30", ".idata ", ".rdata4", ".ajelhf", ".l2", ".ctrjiq", ".oqjrjq", "sosata2", "sosata1", ".estisb", ".333333r", ".999999q", ".mfm", ".?o", "jklsk", "kem", "gqoftmf", ".+{&", ".}^f", ".uma", ".zzp", ".uactz", ".xcpad", "bubd", ".)m=", ".q14", ".eter0", ".eter1", ".i4l", ".=2", "cpvx", "fdfgtrg", ".iqeoz", ".zoihv", ".fal", ".doxozit", ".sexe", ".powina", ".xojap", ". 1d", "._ob", "./lm", ")i7=i", "l(mmx", "6)6@c", ".qqvsqg6", ".idat_0", "xm%2\n tq", ".noyu", ".paqgi", ".rfqzb", "mgl", ".hzcjv", ".t}", ".p5}", ".!(", ".d?u", "ldw", ".ukm", ".;n{", "h5yvi4y4", ".kjcvw", "lines", ".myidata", "8hmhl+\u000e0", "f9c_w-d", "$vvmz=)?", "=ied.7d>", "]!!5pu!*", ":9o>6z-1", "vm<11.23", ">%5?a08;", ".\"dp", ".h~a", ".^1@", ".yif", ".sofetu", ".pom", ".uabw2", ".d}k", "voytzfft", "twvyizui", ".ceceyu", ".tolusu", "nqk", "/94", "/110", ".nmjun", ".cinep", ".wimekog", ".dojores", ".q", ".nuyucw", ".trwqwz", ".:hq", "pwfsbi", ".negena", ".lnplj", "dady", ".dp{", ".nrana", ".yg/", ".fohapi", ".xanip", ".tocoro", ".zewuja", "cjiw", ".8qe3b", ".wofinan", ".gues21", ".nsbyy", "./{z", ".7e.", ".|qy", ".ko_", ".@ri", ".%tp", "rcryptor", " 0", "mklyalsd", "cyqwaajd", ".maxec", "tmr", "kfgei", ".fsvvv", ".k<l", ".0ks", ".?::", "qibqdv", "bogzgdhl", "mnkislbd", ".gduejso", ".()3", "منan/", "\u000b4^\"f", "5\u0005", "b>#m", ".bpua7qt", ".rowkx", "qbltx\u0013ps", ".eeee", "\u0019wmk&qmk", ".datai", "ayt", ".muyicaf", ".cotexep", ".bezetiz", ".sux", ".riy", ".ghdsx", ".roxocij", ".suco", ".pucoro", "perpgszc", "jypgoveg", "hugmhp", ".cpaxiha", "\u001am\u0005n\u001ea\u000e", ".:i", ".!p ", "tecmb", ".\"1t", ".dvb", ".@ut", ".bdata", ".m0tdk", "eoyuwxak", "efdiidnn", "ycrxvds", ".l1", "htskhx"
        ]

        features = {
            "suspicious_sections": 0,
            "num_blacklist_hits": 0,
            "num_entropy_hits": 0,
            "num_fuzzy_hits": 0,
        }

        suspicious_count = 0 
        for section in self.pe.sections: 
            name = section.Name.decode('utf-8', errors='ignore').strip('\x00').lower() 
            
            if name in suspicious_names:
                suspicious_count += 1
                features["num_blacklist_hits"] += 1
                continue
            
            if len(name) >= 5 and shannon_entropy(name) >= 2.0:
                suspicious_count += 1
                features["num_entropy_hits"] += 1
                continue

            distances = [levenshtein(name, bad) for bad in suspicious_names]
            if distances and min(distances) <= 3:
                suspicious_count += 1
                features["num_fuzzy_hits"] += 1
                continue

        features["suspicious_sections"] = suspicious_count
        return features


    def get_dll_characteristics(self):
        chars = getattr(self.pe.OPTIONAL_HEADER, "DllCharacteristics", 0)
        feats = {
            "dll_aslr": 1 if chars & 0x0040 else 0,
            "dll_nx": 1 if chars & 0x0100 else 0,
            "dll_guard": 1 if chars & 0x4000 else 0,
            "dll_high_entropy_va": 1 if chars & 0x0020 else 0,
        }
        feats["dll_chars_total"] = sum(feats.values())
        return feats

    def get_export_info(self):
        num_exports = 0
        try:
            if hasattr(self.pe, "DIRECTORY_ENTRY_EXPORT"):
                num_exports = len(self.pe.DIRECTORY_ENTRY_EXPORT.symbols)
        except Exception:
            pass
        return {
            "num_exports": num_exports,
        }

    def get_resource_info(self):
        feats = {"num_resources": 0, "resource_size_total": 0, "has_version_info": 0, "has_icons": 0}

        def walk(entries, depth=0, seen=0):
            if depth > self.MAX_RESOURCE_DEPTH or seen > self.MAX_RESOURCE_ENTRIES:
                return 0, 0, 0, 0
            count = size = ver = ico = 0
            for res in entries:
                count += 1
                if getattr(res, "id", None) == pefile.RESOURCE_TYPE.get("RT_VERSION"):
                    ver = 1
                if getattr(res, "id", None) == pefile.RESOURCE_TYPE.get("RT_ICON"):
                    ico = 1
                if hasattr(res, "directory") and hasattr(res.directory, "entries"):
                    sub_count, sub_size, sub_ver, sub_ico = walk(res.directory.entries, depth + 1, seen + count)
                    count += sub_count
                    size += sub_size
                    ver |= sub_ver
                    ico |= sub_ico
                if hasattr(res, "data") and hasattr(res.data, "struct"):
                    size += getattr(res.data.struct, "Size", 0)
            return count, size, ver, ico

        try:
            if hasattr(self.pe, "DIRECTORY_ENTRY_RESOURCE"):
                count, size, ver, ico = walk(self.pe.DIRECTORY_ENTRY_RESOURCE.entries)
                feats.update({
                    "num_resources": count,
                    "resource_size_total": size,
                    "has_version_info": ver,
                    "has_icons": ico,
                })
        except Exception:
            pass
        return feats

    def get_tls_info(self):
        feats = {"has_tls": 0, "num_tls_callbacks": 0}
        try:
            if hasattr(self.pe, "DIRECTORY_ENTRY_TLS"):
                feats["has_tls"] = 1
                tls = self.pe.DIRECTORY_ENTRY_TLS.struct
                callback_array = tls.AddressOfCallBacks
                if callback_array:
                    addr = callback_array - self.pe.OPTIONAL_HEADER.ImageBase
                    ptr_size = 8 if self.pe.PE_TYPE == pefile.OPTIONAL_HEADER_MAGIC_PE_PLUS else 4
                    for i in range(self.MAX_TLS_CALLBACKS):
                        try:
                            cb = (self.pe.get_qword_at_rva(addr) if ptr_size == 8
                                  else self.pe.get_dword_at_rva(addr))
                        except Exception:
                            break
                        if cb == 0:
                            break
                        feats["num_tls_callbacks"] += 1
                        addr += ptr_size
        except Exception:
            pass
        feats["num_tls_callbacks"] = feats["num_tls_callbacks"]
        return feats

    def get_section_size_ratios(self):
        ratios = []
        for s in getattr(self.pe, "sections", []):
            raw_size = s.SizeOfRawData
            virt_size = s.Misc_VirtualSize
            if raw_size > 0:
                ratios.append(virt_size / raw_size)
        avg = np.mean(ratios) if ratios else 0
        return {"section_size_ratio_avg": avg}

    def get_alignment_info(self):
        return {
            "image_base": getattr(self.pe.OPTIONAL_HEADER, "ImageBase", 0),
            "section_alignment": getattr(self.pe.OPTIONAL_HEADER, "SectionAlignment", 0),
            "file_alignment": getattr(self.pe.OPTIONAL_HEADER, "FileAlignment", 0),
        }

    def get_debug_info(self):
        size = 0
        try:
            for dbg in getattr(self.pe, "DIRECTORY_ENTRY_DEBUG", []):
                size += getattr(dbg.struct, "SizeOfData", 0)
        except Exception:
            pass
        return {"has_debug": 1 if size > 0 else 0, "debug_size": size}

    # ------------------- Feature aggregation -------------------

    def get_features(self):
        feats = {
            "NumberOfSections": self.get_number_of_sections(),
            "TimeDateStamp": self.get_timedatestamp(),
            "NumberOfSymbols": self.get_number_of_symbols(),
            "MajorLinkerVersion": self.get_major_linker_version(),
            "MinorLinkerVersion": self.get_minor_linker_version(),
            "SizeOfCode": self.get_size_of_code(),
            "SizeOfInitializedData": self.get_size_of_initialized_data(),
            "SizeOfUninitializedData": self.get_size_of_uninitialized_data(),
            "AddressOfEntryPoint": self.get_address_of_entry_point(),  # categorical, not log
            "SizeOfImage": self.get_size_of_image(),
            "Checksum": self.get_checksum(),
            "Subsystem": self.get_subsystem(),  # categorical
            "SuspiciousCalls": self.get_spyware_api_calls(),
            "OverlaySize": self.get_overlay_size()
        }
        feats.update(self.get_score_suspicious_sections())
        feats.update(self.get_alignment_info())
        feats.update(self.get_debug_info())
        feats.update(self.get_section_size_ratios())
        feats.update(self.get_tls_info())
        feats.update(self.get_resource_info())
        feats.update(self.get_export_info())
        feats.update(self.get_dll_characteristics())
        feats.update(self.get_section_anomalies())
        feats.update(self.get_imported_dlls_and_functions())
        for sec in [".rsrc", ".reloc", ".rdata", ".text", ".data", ".bss", ".idata"]:
            feats.update(self.get_section_info(sec))
        return feats
