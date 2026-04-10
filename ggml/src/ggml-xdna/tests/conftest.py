import pytest

try:
    from iron.common import AIEContext
    import aie.utils as aie_utils

    @pytest.fixture
    def aie_context():
        """Create a fresh AIEContext for each test, with cleanup."""
        ctx = AIEContext()
        yield ctx
        aie_utils.DefaultNPURuntime.cleanup()
except ImportError:
    pass
