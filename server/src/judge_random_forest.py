from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from src.judge import Judge

class JudgeRandomForest(Judge):
    def __init__(self):
        super().__init__(feature_list=[
            "NumberOfSections","TimeDateStamp","NumberOfSymbols","MajorLinkerVersion","MinorLinkerVersion","SizeOfCode","SizeOfInitializedData","SizeOfUninitializedData","AddressOfEntryPoint","SizeOfImage","Checksum","Subsystem","SuspiciousCalls","OverlaySize","num_blacklist_hits","num_entropy_hits","num_fuzzy_hits","image_base","section_alignment","file_alignment","has_debug","debug_size","section_size_ratio_avg","has_tls","num_tls_callbacks","num_resources","resource_size_total","has_version_info","has_icons","num_exports","dll_nx","dll_guard","dll_high_entropy_va","dll_chars_total","section_executable","section_writable","section_rwx","sections_entropy_high","ImportedDLLs","ImportedFunctions",".rsrc_exists",".rsrc_SizeOfRawData",".rsrc_entropy",".reloc_exists",".reloc_SizeOfRawData",".reloc_entropy",".rdata_exists",".rdata_SizeOfRawData",".rdata_entropy",".text_exists",".text_SizeOfRawData",".text_entropy",".data_exists",".data_SizeOfRawData",".data_entropy",".bss_exists",".bss_SizeOfRawData",".bss_entropy",".idata_exists",".idata_SizeOfRawData",".idata_entropy"                           
        ])

        self.pipeline = Pipeline([
            ('random_forest', RandomForestClassifier(class_weight={0: 0.7, 1: 1.3}))
        ])

if __name__ == "__main__":
    judge = JudgeRandomForest()
    judge.fit()
    judge.evaluate()
    judge.save_model('./judges/judge_random_forest.joblib')
