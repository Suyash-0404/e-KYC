"""Domain configuration for e-KYC system"""
DOMAINS = {
    "banking": {"name": " Banking & Finance", "face_threshold": 75.0, "required_fields": ["name", "dob", "id_number"], "verification_checks": 3},
    "appstore": {"name": " App Store Purchases", "face_threshold": 70.0, "required_fields": ["name", "id_number"], "verification_checks": 3},
    "gaming": {"name": " Gaming Platforms", "face_threshold": 65.0, "required_fields": ["name"], "verification_checks": 2},
    "restaurant": {"name": "️ Restaurant & Bar", "face_threshold": 60.0, "required_fields": ["name", "dob"], "verification_checks": 2}
}

def get_domain_config(domain_key):
    return DOMAINS.get(domain_key, DOMAINS["restaurant"])

def get_all_domains():
    return DOMAINS

def validate_threshold(threshold):
    return min(threshold, 75.0)
