from aquilify.core.routing import rule, include
import views

# ROUTER configuration.

# The `ROUTER` list routes URLs to views.
# Examples:
# Function views
#     1. Add an import:  from my_app import views
#     2. Add a URL to ROUTER:  rule('/', views.home, name='home')
# Including another ROUTING
#     1. Import the include() function: from aquilify.core.routing import include, rule
#     2. Add a URL to ROUTER:  rule('/blog', include = include('blog.routing'))

ROUTER = [
    rule("/api/v1", include = include("api.routing"), methods = ["GET", "POST"], name = "Analyser | API_V1"),
    rule("/", views.homeview)
    # rule("/api/v1", include = include("api.routing"))
]
