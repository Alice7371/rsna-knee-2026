"""Submit a finished kernel version to a notebook-only competition."""
import sys

from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.competitions.types.competition_api_service import (
    ApiCreateCodeSubmissionRequest,
)


def main(owner, slug, version, competition, message):
    api = KaggleApi()
    api.authenticate()
    with api.build_kaggle_client() as kaggle:
        req = ApiCreateCodeSubmissionRequest()
        req.file_name = "submission.csv"
        req.competition_name = competition
        req._kernel_owner = owner
        req.kernel_slug = slug
        if version:
            req.kernel_version = int(version)
        if message:
            req.submission_description = message
        resp = kaggle.competitions.competition_api_client.create_code_submission(req)
    print("result:", resp.message)
    print("url:", getattr(resp, "url", None))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
