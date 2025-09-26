from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from src.judge import Judge

class JudgeLogisticRegression(Judge):
    def __init__(self):
        super().__init__(feature_list=[
            "TimeDateStamp", "MajorLinkerVersion", "MinorLinkerVersion", "SizeOfCode", "SizeOfInitializedData", "SizeOfUninitializedData", "AddressOfEntryPoint", "SizeOfImage", "Checksum", "SuspiciousCalls", "OverlaySize", "num_entropy_hits", "num_fuzzy_hits", "file_alignment", "has_debug", "section_size_ratio_avg", "has_tls", "num_tls_callbacks", "num_resources", "resource_size_total", "has_version_info", "has_icons", "num_exports", "dll_aslr", "dll_nx", "dll_guard", "section_executable", "section_writable", "sections_entropy_high", "ImportedDLLs", "ImportedFunctions", ".rsrc_exists", ".rsrc_SizeOfRawData", ".rsrc_entropy", ".reloc_entropy", ".rdata_entropy", ".text_SizeOfRawData", ".bss_exists", ".idata_entropy"
        ])

        self.pipeline = Pipeline([
            ('poly', PolynomialFeatures(degree = 2, include_bias = False)),
            ('scaler', StandardScaler()),
            ('logistic_regression', LogisticRegression(class_weight={0: 0.7, 1: 1.3}))
        ])

if __name__ == "__main__":
    judge = JudgeLogisticRegression()
    judge.fit()
    judge.evaluate()
    judge.save_model('./judges/judge_logistic_regression.joblib')