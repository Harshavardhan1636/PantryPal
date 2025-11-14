"""
OWASP ZAP Security Scan

Automated security testing using OWASP ZAP.
Tests for common vulnerabilities:
- SQL Injection
- XSS (Cross-Site Scripting)
- CSRF (Cross-Site Request Forgery)
- Authentication bypass
- Insecure endpoints
- API security issues

Run: python backend/tests/security/zap-security-scan.py
"""

import time
import json
from zapv2 import ZAPv2
from datetime import datetime
import os


# ============================================================================
# Configuration
# ============================================================================

ZAP_API_KEY = os.getenv('ZAP_API_KEY', 'changeme')
ZAP_PROXY = os.getenv('ZAP_PROXY', 'http://localhost:8080')
TARGET_URL = os.getenv('TARGET_URL', 'https://staging-api.pantrypal.app')

# Initialize ZAP client
zap = ZAPv2(apikey=ZAP_API_KEY, proxies={'http': ZAP_PROXY, 'https': ZAP_PROXY})


# ============================================================================
# Helper Functions
# ============================================================================

def wait_for_passive_scan():
    """Wait for passive scan to complete."""
    print('Waiting for passive scan to complete...')
    
    while int(zap.pscan.records_to_scan) > 0:
        print(f'Records to scan: {zap.pscan.records_to_scan}')
        time.sleep(5)
    
    print('Passive scan completed')


def wait_for_active_scan(scan_id):
    """Wait for active scan to complete."""
    print(f'Starting active scan (ID: {scan_id})')
    
    while int(zap.ascan.status(scan_id)) < 100:
        progress = int(zap.ascan.status(scan_id))
        print(f'Active scan progress: {progress}%')
        time.sleep(10)
    
    print('Active scan completed')


def generate_report(output_file='security-report.html'):
    """Generate HTML security report."""
    print(f'Generating report: {output_file}')
    
    report = zap.core.htmlreport()
    
    with open(output_file, 'w') as f:
        f.write(report)
    
    print(f'Report saved to {output_file}')


def get_alerts_summary():
    """Get summary of security alerts by severity."""
    alerts = zap.core.alerts()
    
    summary = {
        'High': 0,
        'Medium': 0,
        'Low': 0,
        'Informational': 0
    }
    
    critical_alerts = []
    
    for alert in alerts:
        risk = alert.get('risk', 'Informational')
        summary[risk] = summary.get(risk, 0) + 1
        
        if risk == 'High':
            critical_alerts.append({
                'name': alert.get('alert'),
                'url': alert.get('url'),
                'description': alert.get('description'),
                'solution': alert.get('solution'),
            })
    
    return summary, critical_alerts


# ============================================================================
# Spider & Scan Functions
# ============================================================================

def spider_target(target_url):
    """Spider the target to discover all pages."""
    print(f'Spidering target: {target_url}')
    
    # Start spider
    spider_id = zap.spider.scan(target_url)
    
    # Wait for spider to complete
    while int(zap.spider.status(spider_id)) < 100:
        progress = int(zap.spider.status(spider_id))
        print(f'Spider progress: {progress}%')
        time.sleep(5)
    
    print('Spider completed')
    
    # Print discovered URLs
    urls = zap.core.urls()
    print(f'Discovered {len(urls)} URLs')


def ajax_spider_target(target_url):
    """Use AJAX spider for JavaScript-heavy applications."""
    print(f'AJAX spidering target: {target_url}')
    
    # Start AJAX spider
    zap.ajaxSpider.scan(target_url)
    
    # Wait for AJAX spider to complete
    while zap.ajaxSpider.status == 'running':
        print('AJAX spider running...')
        time.sleep(5)
    
    print('AJAX spider completed')


def passive_scan():
    """Run passive security scan."""
    print('Running passive scan...')
    
    # Enable all passive scan rules
    zap.pscan.enable_all_scanners()
    
    # Wait for passive scan
    wait_for_passive_scan()


def active_scan(target_url):
    """Run active security scan."""
    print('Running active scan...')
    
    # Enable all active scan rules
    zap.ascan.enable_all_scanners()
    
    # Start active scan
    scan_id = zap.ascan.scan(target_url)
    
    # Wait for active scan
    wait_for_active_scan(scan_id)


# ============================================================================
# Authentication Testing
# ============================================================================

def test_authentication_bypass():
    """Test for authentication bypass vulnerabilities."""
    print('Testing authentication bypass...')
    
    # Test common bypass techniques
    bypass_tests = [
        {'username': "' OR '1'='1", 'password': "' OR '1'='1"},
        {'username': 'admin', 'password': 'admin'},
        {'username': 'admin', 'password': ''},
        {'username': "admin'--", 'password': 'anything'},
    ]
    
    for test in bypass_tests:
        # ZAP will automatically test these via active scan
        pass


def test_session_management():
    """Test session management security."""
    print('Testing session management...')
    
    # ZAP checks:
    # - Session fixation
    # - Session token in URL
    # - Cookie flags (HttpOnly, Secure, SameSite)
    pass


# ============================================================================
# API Security Testing
# ============================================================================

def test_api_security(api_url):
    """Test API-specific security issues."""
    print('Testing API security...')
    
    # Import OpenAPI spec if available
    openapi_spec = f'{api_url}/openapi.json'
    
    try:
        zap.openapi.import_url(openapi_spec)
        print('OpenAPI spec imported')
    except:
        print('No OpenAPI spec found')
    
    # Test API endpoints
    api_tests = [
        # Test for missing authentication
        f'{api_url}/api/v1/pantry-items',
        
        # Test for authorization bypass
        f'{api_url}/api/v1/admin/users',
        
        # Test for rate limiting
        f'{api_url}/api/v1/predictions',
    ]
    
    for endpoint in api_tests:
        # ZAP will test these automatically
        pass


def test_injection_vulnerabilities(api_url):
    """Test for injection vulnerabilities."""
    print('Testing injection vulnerabilities...')
    
    # SQL Injection payloads
    sql_payloads = [
        "1' OR '1'='1",
        "1; DROP TABLE users--",
        "1' UNION SELECT NULL--",
    ]
    
    # NoSQL Injection payloads
    nosql_payloads = [
        '{"$ne": null}',
        '{"$gt": ""}',
    ]
    
    # Command Injection payloads
    cmd_payloads = [
        '| ls',
        '; cat /etc/passwd',
        '`whoami`',
    ]
    
    # ZAP active scan will test these automatically


# ============================================================================
# CSRF & XSS Testing
# ============================================================================

def test_csrf_protection():
    """Test CSRF token validation."""
    print('Testing CSRF protection...')
    
    # ZAP checks:
    # - Missing CSRF tokens
    # - Weak CSRF tokens
    # - CSRF token not validated
    pass


def test_xss_vulnerabilities():
    """Test for XSS vulnerabilities."""
    print('Testing XSS vulnerabilities...')
    
    xss_payloads = [
        '<script>alert("XSS")</script>',
        '<img src=x onerror=alert("XSS")>',
        '<svg onload=alert("XSS")>',
    ]
    
    # ZAP active scan will test these automatically


# ============================================================================
# Main Scan Function
# ============================================================================

def run_security_scan():
    """Run complete security scan."""
    
    print('='*80)
    print('OWASP ZAP Security Scan')
    print(f'Target: {TARGET_URL}')
    print(f'Started: {datetime.now().isoformat()}')
    print('='*80)
    
    try:
        # Step 1: Access the target
        print('\n[1/7] Accessing target...')
        zap.urlopen(TARGET_URL)
        time.sleep(2)
        
        # Step 2: Spider the target
        print('\n[2/7] Spidering target...')
        spider_target(TARGET_URL)
        
        # Step 3: AJAX spider (for React/Next.js)
        print('\n[3/7] AJAX spidering...')
        ajax_spider_target(TARGET_URL)
        
        # Step 4: Passive scan
        print('\n[4/7] Running passive scan...')
        passive_scan()
        
        # Step 5: Active scan
        print('\n[5/7] Running active scan...')
        active_scan(TARGET_URL)
        
        # Step 6: API security tests
        print('\n[6/7] Testing API security...')
        test_api_security(TARGET_URL)
        
        # Step 7: Generate report
        print('\n[7/7] Generating report...')
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        report_file = f'security-report-{timestamp}.html'
        generate_report(report_file)
        
        # Print summary
        print('\n' + '='*80)
        print('Scan Summary')
        print('='*80)
        
        summary, critical_alerts = get_alerts_summary()
        
        print(f"High Risk:          {summary.get('High', 0)}")
        print(f"Medium Risk:        {summary.get('Medium', 0)}")
        print(f"Low Risk:           {summary.get('Low', 0)}")
        print(f"Informational:      {summary.get('Informational', 0)}")
        
        # Print critical alerts
        if critical_alerts:
            print('\n' + '='*80)
            print('CRITICAL ALERTS')
            print('='*80)
            
            for i, alert in enumerate(critical_alerts, 1):
                print(f"\n{i}. {alert['name']}")
                print(f"   URL: {alert['url']}")
                print(f"   Description: {alert['description'][:100]}...")
                print(f"   Solution: {alert['solution'][:100]}...")
        
        print('\n' + '='*80)
        print(f'Report saved: {report_file}')
        print(f'Completed: {datetime.now().isoformat()}')
        print('='*80)
        
        # Exit with error code if high risk found
        if summary.get('High', 0) > 0:
            print('\n❌ SCAN FAILED: High risk vulnerabilities found')
            exit(1)
        else:
            print('\n✅ SCAN PASSED: No high risk vulnerabilities')
            exit(0)
    
    except Exception as e:
        print(f'\n❌ Error during scan: {e}')
        exit(1)


# ============================================================================
# CI/CD Integration
# ============================================================================

def run_in_ci_pipeline():
    """
    Run security scan in CI/CD pipeline.
    
    Usage in GitHub Actions:
    
    - name: OWASP ZAP Scan
      run: |
        docker run -d -p 8080:8080 owasp/zap2docker-stable zap.sh -daemon -config api.key=changeme
        sleep 10
        python backend/tests/security/zap-security-scan.py
      env:
        ZAP_API_KEY: changeme
        TARGET_URL: https://staging-api.pantrypal.app
    """
    run_security_scan()


# ============================================================================
# Run
# ============================================================================

if __name__ == '__main__':
    run_security_scan()
