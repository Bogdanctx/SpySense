from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from src.judge import Judge

class JudgeSVM(Judge):
    def __init__(self):
        super().__init__(feature_list=[
            "NumberOfSections","TimeDateStamp","MajorLinkerVersion","MinorLinkerVersion","SizeOfCode","SizeOfInitializedData","SizeOfUninitializedData","AddressOfEntryPoint","SizeOfImage","Checksum","Subsystem","SuspiciousCalls","OverlaySize","suspicious_sections","num_blacklist_hits","num_entropy_hits","num_fuzzy_hits","image_base","section_alignment","file_alignment","has_debug","debug_size","section_size_ratio_avg","has_tls","num_tls_callbacks","num_resources","resource_size_total","has_version_info","has_icons","num_exports","dll_aslr","dll_nx","dll_guard","dll_high_entropy_va","dll_chars_total","section_executable","section_writable","section_rwx","sections_entropy_high","ImportedDLLs","ImportedFunctions",".rsrc_SizeOfRawData",".rsrc_entropy",".reloc_exists",".reloc_SizeOfRawData",".reloc_entropy",".rdata_exists",".rdata_SizeOfRawData",".rdata_entropy",".text_exists",".text_SizeOfRawData",".text_entropy",".data_exists",".data_SizeOfRawData",".data_entropy",".bss_exists",".bss_SizeOfRawData",".bss_entropy",".idata_exists",".idata_SizeOfRawData",".idata_entropy"
        ])

        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('svm', SVC(class_weight = {0: 0.7, 1: 1.3}, probability=True))
        ])

if __name__ == "__main__":
    judge = JudgeSVM()
    judge.fit()
    judge.evaluate()
    judge.save_model('./judges/judge_svm.joblib')