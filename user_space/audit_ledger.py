#!/usr/bin/python3
"""
NetShield-MARL: Phase 4 Cryptographic Audit Ledger
Module Path: user_space/audit_ledger.py

Provides a local, zero-budget, tamper-evident security audit log using a
Cryptographically Chained SQLite Database with SHA-256 sequential hashing.
"""

import sqlite3
import hashlib
import json
import os
from datetime import datetime, timezone
from typing import Tuple, Dict, Any


class CryptographicAuditLedger:
    """
    Cryptographically Chained SQLite Security Audit Ledger.
    Links each security audit log entry (block) to the preceding block using SHA-256 sequential hashing.
    """
    
    GENESIS_PREV_HASH: str = "0" * 64
    
    def __init__(self, db_path: str = "netshield_audit.db"):
        """
        Initialize the audit ledger and ensure the database and genesis block exist.
        
        Args:
            db_path (str): File path to the SQLite database file.
        """
        self.db_path = db_path
        self._init_database()

    def _get_connection(self) -> sqlite3.Connection:
        """Helper to create SQLite connection with pragmas enabled."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_database(self) -> None:
        """
        Creates the 'audit_ledger' table if it does not exist and inserts
        the Genesis Block (Block 0) if the table is empty.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS audit_ledger (
                    block_index INTEGER PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    event_payload TEXT NOT NULL,
                    previous_hash TEXT NOT NULL,
                    current_hash TEXT NOT NULL
                )
            """)
            conn.commit()

            # Check if ledger is empty; if so, create Genesis Block (Block 0)
            cursor.execute("SELECT COUNT(*) FROM audit_ledger")
            if cursor.fetchone()[0] == 0:
                self._create_genesis_block(cursor)
                conn.commit()

    def _create_genesis_block(self, cursor: sqlite3.Cursor) -> None:
        """Internal helper to construct and insert Block 0."""
        genesis_index = 0
        timestamp = datetime.now(timezone.utc).isoformat()
        genesis_payload = {
            "event": "GENESIS_BLOCK",
            "system": "NetShield-MARL",
            "description": "Cryptographic Audit Ledger Initialized"
        }
        payload_json = json.dumps(genesis_payload, sort_keys=True)
        prev_hash = self.GENESIS_PREV_HASH
        
        current_hash = self._calculate_block_hash(
            index=genesis_index,
            timestamp=timestamp,
            payload_json=payload_json,
            prev_hash=prev_hash
        )

        cursor.execute("""
            INSERT INTO audit_ledger (block_index, timestamp, event_payload, previous_hash, current_hash)
            VALUES (?, ?, ?, ?, ?)
        """, (genesis_index, timestamp, payload_json, prev_hash, current_hash))
        
        print(f"🔒 [Audit Ledger] Genesis Block #0 created. Hash: {current_hash[:16]}...")

    @staticmethod
    def _calculate_block_hash(index: int, timestamp: str, payload_json: str, prev_hash: str) -> str:
        """
        Calculates SHA-256 hash over block index, timestamp, JSON payload string, and previous block hash.
        
        Formula: SHA-256( Index + Timestamp + Event_Payload_JSON + Previous_Hash )
        """
        raw_payload = f"{index}{timestamp}{payload_json}{prev_hash}"
        return hashlib.sha256(raw_payload.encode('utf-8')).hexdigest()

    def append_event(self, event_payload: Dict[str, Any]) -> str:
        """
        Computes SHA-256 block hash and inserts a new security log entry atomically.

        Args:
            event_payload (dict): Security event details (source_ip, dest_ip, detected_anomaly_score, action_taken, etc.)

        Returns:
            str: SHA-256 current_hash of the newly appended block.
        """
        payload_json = json.dumps(event_payload, sort_keys=True)
        timestamp = datetime.now(timezone.utc).isoformat()

        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Retrieve previous block details
            cursor.execute("""
                SELECT block_index, current_hash 
                FROM audit_ledger 
                ORDER BY block_index DESC 
                LIMIT 1
            """)
            last_block = cursor.fetchone()
            
            new_index = last_block['block_index'] + 1
            prev_hash = last_block['current_hash']

            # Compute block hash
            curr_hash = self._calculate_block_hash(
                index=new_index,
                timestamp=timestamp,
                payload_json=payload_json,
                prev_hash=prev_hash
            )

            # Insert block atomically
            cursor.execute("""
                INSERT INTO audit_ledger (block_index, timestamp, event_payload, previous_hash, current_hash)
                VALUES (?, ?, ?, ?, ?)
            """, (new_index, timestamp, payload_json, prev_hash, curr_hash))
            
            conn.commit()

        print(f"📝 [Audit Ledger] Block #{new_index} committed | Event: {event_payload.get('action_taken', 'EVENT')} | Hash: {curr_hash[:16]}...")
        return curr_hash

    def verify_chain_integrity(self) -> Tuple[bool, int]:
        """
        Iterates through all blocks in sequential order, recalculates every SHA-256 hash,
        and verifies hash link integrity.

        Returns:
            Tuple[bool, int]: (True, -1) if chain is 100% valid.
                              (False, tampered_block_index) if tampering/corruption is detected.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT block_index, timestamp, event_payload, previous_hash, current_hash 
                FROM audit_ledger 
                ORDER BY block_index ASC
            """)
            blocks = cursor.fetchall()

        if not blocks:
            print("⚠️ [Audit Ledger] Ledger is empty!")
            return False, 0

        expected_prev_hash = self.GENESIS_PREV_HASH

        for idx, block in enumerate(blocks):
            b_index = block['block_index']
            b_timestamp = block['timestamp']
            b_payload = block['event_payload']
            b_prev_hash = block['previous_hash']
            b_curr_hash = block['current_hash']

            # 1. Verify strict sequential index
            if b_index != idx:
                print(f"❌ [TAMPER DETECTED] Index sequence broken at Row #{idx} (Found index {b_index})")
                return False, b_index

            # 2. Verify previous hash matches prior block's current hash
            if b_prev_hash != expected_prev_hash:
                print(f"❌ [TAMPER DETECTED] Block #{b_index}: Previous hash link mismatch!")
                print(f"   Expected: {expected_prev_hash}")
                print(f"   Found:    {b_prev_hash}")
                return False, b_index

            # 3. Recalculate SHA-256 hash over block contents
            recalculated_hash = self._calculate_block_hash(
                index=b_index,
                timestamp=b_timestamp,
                payload_json=b_payload,
                prev_hash=b_prev_hash
            )

            # 4. Verify stored current hash matches recalculated hash
            if recalculated_hash != b_curr_hash:
                print(f"❌ [TAMPER DETECTED] Block #{b_index}: Data content tampered or corrupted!")
                print(f"   Stored Hash:       {b_curr_hash}")
                print(f"   Recalculated Hash: {recalculated_hash}")
                return False, b_index

            # Update expected previous hash for next iteration
            expected_prev_hash = b_curr_hash

        print(f"✅ [Audit Ledger] Chain Integrity Verified: All {len(blocks)} blocks cryptographically valid.")
        return True, -1


if __name__ == "__main__":
    print("=" * 70)
    print("🛡️ NetShield-MARL Phase 4: Cryptographic Audit Ledger Test Suite")
    print("=" * 70)

    test_db_path = "test_netshield_audit.db"
    
    # Remove existing test DB if present
    if os.path.exists(test_db_path):
        os.remove(test_db_path)

    try:
        # Step 1: Initialize Ledger & Genesis Block
        ledger = CryptographicAuditLedger(db_path=test_db_path)

        # Step 2: Append Security Audit Events
        print("\n--- Inserting Valid Security Events ---")
        event1 = {
            "source_ip": "172.18.0.4",
            "dest_ip": "172.18.0.3",
            "detected_anomaly_score": 0.9421,
            "action_taken": "ISOLATE_IPTABLES"
        }
        hash1 = ledger.append_event(event1)

        event2 = {
            "source_ip": "172.18.0.5",
            "dest_ip": "172.18.0.2",
            "detected_anomaly_score": 0.7810,
            "action_taken": "RATE_LIMIT"
        }
        hash2 = ledger.append_event(event2)

        # Step 3: Run Initial Integrity Audit
        print("\n--- Running Initial Chain Integrity Audit ---")
        is_valid, bad_block = ledger.verify_chain_integrity()
        assert is_valid is True and bad_block == -1, "Initial integrity check failed!"

        # Step 4: Simulate Malicious Database Tampering
        print("\n--- Simulating Malicious Data Modification (Attacker Modifies Block #1) ---")
        conn = sqlite3.connect(test_db_path)
        cursor = conn.cursor()
        
        # Modify payload in Block #1 directly via raw SQL
        tampered_payload = json.dumps({
            "source_ip": "172.18.0.4",
            "dest_ip": "172.18.0.3",
            "detected_anomaly_score": 0.1000,  # Attacker lowered score to hide attack!
            "action_taken": "MONITOR"
        }, sort_keys=True)
        
        cursor.execute("UPDATE audit_ledger SET event_payload = ? WHERE block_index = 1", (tampered_payload,))
        conn.commit()
        conn.close()
        print("⚠️ Malicious modification applied to Block #1 payload in SQLite file.")

        # Step 5: Verify Tamper Detection
        print("\n--- Running Integrity Audit Post-Tampering ---")
        is_valid_after_tamper, tampered_index = ledger.verify_chain_integrity()
        
        if not is_valid_after_tamper and tampered_index == 1:
            print(f"\n🎉 TEST SUCCESS: Cryptographic chain successfully caught tampering at Block #{tampered_index}!")
        else:
            raise RuntimeError("❌ TEST FAILED: Tamper detection suite failed to detect modified block!")

    finally:
        # Cleanup test DB file
        if os.path.exists(test_db_path):
            os.remove(test_db_path)
            print("\n🧹 Cleaned up temporary test database.")
