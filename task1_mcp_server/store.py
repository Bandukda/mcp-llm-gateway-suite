"""In-memory billing store standing in for the real CRM/ledger.

Everything here is deliberately boring. It exists so the tool handlers have
something to succeed and fail against, and so the difference between a
*protocol* error and a *business* error is demonstrable:

  * "CUST-9" is not a customer id            -> protocol error, -32602
  * "CUST-99999" is a well-formed id we have -> business error, isError result
    no record for
"""

from __future__ import annotations

import hashlib
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone


class CustomerNotFoundError(LookupError):
    """A well-formed customer id that does not exist."""


class RefundNotPermittedError(ValueError):
    """A well-formed refund that business rules reject."""


class IdempotencyKeyConflict(ValueError):
    """The key was reused with a different request."""


@dataclass
class Customer:
    customer_id: str
    name: str
    email: str
    plan: str
    status: str
    lifetime_value_usd: float
    refundable_balance_usd: float


@dataclass
class Refund:
    refund_id: str
    customer_id: str
    amount_usd: float
    reason: str
    status: str
    created_at: str
    idempotency_key: str | None = None
    request_fingerprint: str | None = None


@dataclass
class BillingStore:
    customers: dict[str, Customer] = field(default_factory=dict)
    refunds: list[Refund] = field(default_factory=list)
    _by_idempotency_key: dict[str, Refund] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def get_customer(self, customer_id: str) -> Customer:
        try:
            return self.customers[customer_id]
        except KeyError:
            raise CustomerNotFoundError(f"No customer record for {customer_id}") from None

    @staticmethod
    def _fingerprint(customer_id: str, amount_usd: float, reason: str) -> str:
        """What the idempotency key is a promise about."""
        return hashlib.sha256(
            f"{customer_id}|{amount_usd:.2f}|{reason}".encode("utf-8")
        ).hexdigest()

    def create_refund(
        self,
        customer_id: str,
        amount_usd: float,
        reason: str,
        idempotency_key: str | None = None,
    ) -> tuple[Refund, bool]:
        """Create a refund. Returns ``(refund, was_replayed)``."""
        fingerprint = self._fingerprint(customer_id, amount_usd, reason)

        with self._lock:
            if idempotency_key and idempotency_key in self._by_idempotency_key:
                existing = self._by_idempotency_key[idempotency_key]
                # The key has to be bound to the request it was issued for.
                # Without this check, replaying a key with *different* arguments
                # returned the original refund and reported ok/replayed -- so a
                # call asking to refund CUST-20099 $5 was answered with
                # CUST-10042's $1000 refund, the $5 never happened, and the
                # caller was told it had succeeded. Stripe and friends return a
                # 4xx on key reuse with a changed payload; this is that.
                if existing.request_fingerprint != fingerprint:
                    raise IdempotencyKeyConflict(
                        f"idempotency_key {idempotency_key!r} was already used for a "
                        "different refund request; use a new key or resend the "
                        "original arguments"
                    )
                return existing, True

            customer = self.get_customer(customer_id)
            if customer.status != "active":
                raise RefundNotPermittedError(
                    f"Customer {customer_id} has status '{customer.status}'; refunds are only "
                    "permitted for active customers"
                )
            if amount_usd > customer.refundable_balance_usd:
                raise RefundNotPermittedError(
                    f"Refund of {amount_usd:.2f} USD exceeds the refundable balance of "
                    f"{customer.refundable_balance_usd:.2f} USD for {customer_id}"
                )

            refund = Refund(
                refund_id=f"RF-{uuid.uuid4().hex[:12].upper()}",
                customer_id=customer_id,
                amount_usd=round(amount_usd, 2),
                reason=reason,
                status="pending_settlement",
                created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                idempotency_key=idempotency_key,
                request_fingerprint=fingerprint,
            )
            customer.refundable_balance_usd = round(
                customer.refundable_balance_usd - refund.amount_usd, 2
            )
            self.refunds.append(refund)
            if idempotency_key:
                self._by_idempotency_key[idempotency_key] = refund
            return refund, False


def seeded_store() -> BillingStore:
    store = BillingStore()
    for customer in (
        Customer("CUST-10042", "Ada Lovelace", "ada@example.com", "enterprise", "active", 48250.00, 1200.00),
        Customer("CUST-20099", "Grace Hopper", "grace@example.com", "pro", "active", 7310.50, 150.00),
        Customer("CUST-30007", "Alan Turing", "alan@example.com", "starter", "suspended", 210.00, 0.00),
    ):
        store.customers[customer.customer_id] = customer
    return store
