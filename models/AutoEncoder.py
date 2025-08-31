import torch
import torch.nn as nn
import torch.nn.functional as F

class HybridLoss(nn.Module):
    """
    Hàm loss lai kết hợp L1Loss và CosineSimilarityLoss bằng phép nhân.
    Công thức: Loss = L1Loss * (1 + CosineSimilarityLoss)
    """
    def __init__(self):
        super(HybridLoss, self).__init__()
        self.l1_loss = nn.L1Loss(reduction='mean')

    def forward(self, reconstructed_vectors, original_vectors):
        # 1. Tính L1 Loss (Mean Absolute Error)
        # Đo lường sự khác biệt về độ lớn, ít nhạy cảm với outlier
        l1 = self.l1_loss(reconstructed_vectors, original_vectors)

        # 2. Tính Cosine Similarity Loss
        # Đo lường sự khác biệt về hướng (góc độ)
        # F.cosine_similarity trả về giá trị trong khoảng [-1, 1]
        cosine_sim = F.cosine_similarity(reconstructed_vectors, original_vectors, dim=1)
        # Chuyển thành loss trong khoảng [0, 2]
        # Chúng ta tính trung bình loss trên toàn bộ batch
        cosine_loss = (1 - cosine_sim).mean()

        # 3. Kết hợp theo công thức của bạn
        # (1 + cosine_loss) sẽ nằm trong khoảng [1, 3]
        # Nó hoạt động như một hệ số nhân trừng phạt cho L1 loss
        total_loss = l1 * (1 + cosine_loss)

        return total_loss

# =============================================================================
#  KIẾN TRÚC AUTOENCODER
# =============================================================================
class VectorAutoencoder(nn.Module):
    """
    Mô hình Autoencoder để nén và giải nén vector embedding.
    Bao gồm Dropout để chống overfitting.
    """
    def __init__(self, input_dim=768, compressed_dim=101, dropout_rate=0.1):
        super(VectorAutoencoder, self).__init__()

        # --- PHẦN NÉN (ENCODER) ---
        # Nhận vector gốc, nén xuống kích thước nhỏ hơn
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 384),
            nn.ReLU(),
            nn.Dropout(dropout_rate), # Lớp Dropout chống overfitting
            nn.Linear(384, compressed_dim)
        )

        # --- PHẦN GIẢI NÉN (DECODER) ---
        # Nhận vector đã nén, cố gắng tái tạo lại vector gốc
        self.decoder = nn.Sequential(
            nn.Linear(compressed_dim, 384),
            nn.ReLU(),
            nn.Dropout(dropout_rate), # Lớp Dropout chống overfitting
            nn.Linear(384, input_dim)
        )

    def forward(self, x):
        # x là batch các vector gốc
        compressed = self.encoder(x)
        reconstructed = self.decoder(compressed)
        return reconstructed

    def compress(self, x):
        # Một hàm tiện ích để chỉ thực hiện việc nén
        return self.encoder(x)