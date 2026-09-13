import json
import os
from typing import List, Dict, Any
from datetime import datetime

RAW_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "raw-data")

# ---------------------------------------------------------------------------
# Tool #1: search_product_catalog
# Đọc file product_catalog.json, lọc theo category và max_price.
# ---------------------------------------------------------------------------

def search_product_catalog(category: str, max_price: int = 999999999999) -> List[Dict[str, Any]]:
    """
    Tra cứu sản phẩm/dịch vụ Vingroup theo danh mục và giá tối đa.
    
    Args:
        category: Loại sản phẩm ('xe_dien' hoặc 'du_lich').
        max_price: Giá tối đa (VNĐ). Mặc định không giới hạn.
    
    Returns:
        Danh sách sản phẩm phù hợp điều kiện.
    """
    catalog_file = os.path.join(RAW_DATA_DIR, "product_catalog.json")
    if not os.path.exists(catalog_file):
        return [{"error": "Product catalog file not found."}]

    with open(catalog_file, "r", encoding="utf-8") as f:
        products = json.load(f)

    # Chuẩn hoá đầu vào: category không phân biệt hoa/thường, max_price ép về int
    category = (category or "").strip().lower()
    try:
        max_price = int(max_price)
    except (TypeError, ValueError):
        max_price = 999999999999

    results = [
        p for p in products
        if p.get("category", "").lower() == category
        and p.get("price_vnd", 0) <= max_price
    ]
    # Sắp xếp theo giá tăng dần để Agent trình bày dễ đọc hơn
    results.sort(key=lambda p: p.get("price_vnd", 0))
    return results


# ---------------------------------------------------------------------------
# Tool #2: submit_support_ticket
# Tạo ticket mới và lưu (append) vào support_tickets.json.
# ---------------------------------------------------------------------------

def submit_support_ticket(
    customer_name: str,
    issue_description: str,
    priority: str = "medium"
) -> Dict[str, Any]:
    """
    Ghi nhận yêu cầu hỗ trợ của khách hàng vào hệ thống ticket.
    
    Args:
        customer_name: Tên khách hàng.
        issue_description: Mô tả vấn đề cần hỗ trợ.
        priority: Mức độ ưu tiên ('low', 'medium', 'high'). Mặc định 'medium'.
    
    Returns:
        Thông tin ticket vừa tạo bao gồm ticket_id, status.
    """
    tickets_file = os.path.join(RAW_DATA_DIR, "support_tickets.json")

    # Chuẩn hoá priority — chỉ chấp nhận 3 mức, mặc định 'medium'
    priority = (priority or "medium").strip().lower()
    if priority not in ("low", "medium", "high"):
        priority = "medium"

    # Load existing tickets (tránh Trap 2: ghi đè thay vì append)
    existing_tickets: List[Dict[str, Any]] = []
    if os.path.exists(tickets_file):
        with open(tickets_file, "r", encoding="utf-8") as f:
            try:
                existing_tickets = json.load(f)
            except json.JSONDecodeError:
                existing_tickets = []

    # Generate ticket ID: TK-YYYYMMDD-SEQ
    now = datetime.now()
    today = now.strftime("%Y%m%d")
    seq = len(existing_tickets) + 1
    ticket_id = f"TK-{today}-{seq:03d}"

    new_ticket = {
        "ticket_id": ticket_id,
        "customer_name": customer_name,
        "issue_description": issue_description,
        "priority": priority,
        "status": "open",
        "created_at": now.isoformat() + "+07:00",
        "category": "general",
    }
    existing_tickets.append(new_ticket)

    os.makedirs(os.path.dirname(tickets_file), exist_ok=True)
    with open(tickets_file, "w", encoding="utf-8") as f:
        json.dump(existing_tickets, f, indent=2, ensure_ascii=False)

    return {
        "ticket_id": ticket_id,
        "customer_name": customer_name,
        "issue_description": issue_description,
        "priority": priority,
        "status": "open",
        "message": f"Ticket {ticket_id} đã được tạo thành công.",
    }


# ---------------------------------------------------------------------------
# TOOL_DEFINITIONS — JSON Schemas mô tả cho LLM
# ---------------------------------------------------------------------------

TOOL_DEFINITIONS = [
    {
        "name": "search_product_catalog",
        "description": (
            "Tra cứu sản phẩm/dịch vụ Vingroup (xe điện VinFast, kỳ nghỉ Vinpearl) "
            "theo danh mục và giá tối đa. Dùng khi khách hàng muốn xem, tìm, so sánh "
            "hoặc hỏi giá sản phẩm."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "description": "Loại sản phẩm: 'xe_dien' (xe điện VinFast) hoặc 'du_lich' (nghỉ dưỡng Vinpearl).",
                    "enum": ["xe_dien", "du_lich"],
                },
                "max_price": {
                    "type": "integer",
                    "description": "Giá tối đa tính bằng VNĐ (ví dụ 600 triệu = 600000000). Bỏ trống nếu không giới hạn.",
                    "minimum": 0,
                },
            },
            "required": ["category"],
        },
    },
    {
        "name": "submit_support_ticket",
        "description": (
            "Ghi nhận yêu cầu hỗ trợ / khiếu nại / phản hồi của khách hàng vào hệ thống ticket. "
            "Dùng khi khách hàng báo lỗi sản phẩm, gặp sự cố dịch vụ hoặc muốn được liên hệ hỗ trợ."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "customer_name": {
                    "type": "string",
                    "description": "Họ tên khách hàng (lấy từ câu nói của khách, ví dụ 'Tôi tên Lê Minh Khoa').",
                },
                "issue_description": {
                    "type": "string",
                    "description": "Mô tả ngắn gọn, rõ ràng vấn đề khách hàng gặp phải.",
                },
                "priority": {
                    "type": "string",
                    "description": "Mức độ ưu tiên: 'high' nếu khách nói gấp/nghiêm trọng, 'low' nếu không gấp, còn lại 'medium'.",
                    "enum": ["low", "medium", "high"],
                    "default": "medium",
                },
            },
            "required": ["customer_name", "issue_description"],
        },
    },
]


# ---------------------------------------------------------------------------
# TOOL_MAP — Ánh xạ tên tool → hàm thực thi
# ---------------------------------------------------------------------------

TOOL_MAP = {
    "search_product_catalog": search_product_catalog,
    "submit_support_ticket": submit_support_ticket
}
